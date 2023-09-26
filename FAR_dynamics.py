import os
import multiprocessing
import re
import argparse
from operator import itemgetter
import itertools
from rdkit import Chem
from rdkit.Chem import AllChem

program_description = "Runs FAR protocol"
parser = argparse.ArgumentParser(description=program_description)

parser.add_argument("-l", "--ligand", type=str,
                    help="Ligand file or folder containing multiple ligands in SDF format", nargs='?')
parser.add_argument("-p", "--peptide", type=str,
                    help="Peptide file or folder containing multiple peptides in PDB format", nargs='?')
parser.add_argument("-r", "--receptor", type=str,
                    help="Receptor file or folder containing multiple receptors in PDB format", nargs='?')
parser.add_argument("-GPU", "--GPU_range", type=str,
                    help="Specify GPUs from 0 to 7 (e.g: 0-3 to utilize GPUs 0,1,2)", required=True)
parser.add_argument("-in", "--in_folder", type=str,
                    help="Specify folder where all config files are located (e.g: /home/user/FAR_and_prod_in_files)", required=True)
parser.add_argument("-t", "--threads", type=int,
                    help="Number of threads to utilize", required=True)
parser.add_argument("-co", "--cofactor", nargs='?',
                    help="Optional: Cofactor file in SDF format or folder containing multiple cofactors (if folder RECEPTOR'S FILENAME WITHOUT EXTENSION MUST BE WITHIN COFACTOR'S FILE NAME)")
parser.add_argument("-cop", "--cofactor_prefix", nargs='?',
                    help="if you provided a cofactor's folder you must enter a prefix that will be used to find the file (e.g.: -cop SAM_ ---will-find--> SAM_receptorName.sdf)")
parser.add_argument("-prod", "--prod", nargs='?',
                    help="Optional: Runs given nano seconds of molecular dynamics production phase and utilizes MMPBSA for calculation of binding free energy (e.g.: -prod 5 ---> will run 5ns)")
parser.add_argument("-prod_only", "--prod_only", nargs='?',
                    help="Optional: Only runs molecular dynamics production phase and utilizes MMPBSA for calculation of binding free energy (MUST BE RUNED IN DIRECTORY WHERE PREVIOUSLY FAR WAS PERFORMED)")
args = parser.parse_args()
prod_only = args.prod_only
if not prod_only:
    ligand_type = args.ligand
    peptide_type = args.peptide
    if ligand_type:
        if "sdf" in ligand_type:
            ligand = os.path.abspath(args.ligand)
        else:
            parser.error(f"You must provide ligand in sdf format")
    if peptide_type:
        if "pdb" in peptide_type:
            ligand = os.path.abspath(args.peptide)
        else:
            parser.error(f"You must provide peptide in pdb format")
    if not ligand_type and not peptide_type:
        parser.error(f"You must provide ligand (-l) or peptide (-p)")
    receptor = args.receptor
    if receptor:  
        receptor = os.path.abspath(args.receptor)
    if not receptor:
        parser.error(f"You must provide receptor (-r)")  
    complex_file = f'{ligand.replace(".sdf", "")}'
    cofactor_folder = args.cofactor
    prefix_cofactor = args.cofactor_prefix
    prod = args.prod
    time_prod = prod
    if cofactor_folder:
        cofactor_folder = os.path.abspath(cofactor_folder)
        if prefix_cofactor == None and os.path.isdir(cofactor_folder):
            parser.error(f"if you provided a cofactor's folder you must enter a prefix that will be used to find the files (e.g.: -cop SAM_ ---will-find--> SAM_receptorName.sdf)")
if prod_only:
    time_prod = prod_only
threads = args.threads
range_GPU = args.GPU_range
in_folder = os.path.abspath(args.in_folder)
range_GPU_S = int(range_GPU.split("-")[0])
range_GPU_E = int(range_GPU.split("-")[1])



if not prod_only:
    if not prod:
        print("Running FAR protocol")
    if prod:
        threads == len(range(range_GPU_S,range_GPU_E))
        print(f"Running FAR protocol and {time_prod}ns of molecular dynamics production with MMPBSA calculation")

if prod_only:
    print(f"Running only {time_prod}ns of molecular dynamics production with MMPBSA calculation")

def data_seeker(file, startswith, mode):
    if mode == "atoms_number":
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith(startswith):
                    data = line.split()[3]
                    return data
                
    if mode == "residues_number":
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                if startswith in line:
                    data = line.split()[0]
                    return data
                
    if mode == "residues_number_prot":
        mol_line_list = []
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                mol_line = re.findall('^[0-9].*', line, re.MULTILINE)
                if mol_line:
                    mol_line_list.append(line)

        line = mol_line_list[-2]
        end_res_prot = int(line.split()[4])
        return end_res_prot
        
    if mode == "atoms_data":
        end_atom_1_out = []
        end_atom2_out = []
        mol_line_list = []
        start_end_atomi_out = []
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                mol_line = re.findall('^[0-9].*', line, re.MULTILINE)
                if mol_line:
                    mol_line_list.append(line)
        data = []
        for i in range(len(mol_line_list)):
            line = mol_line_list[i]
            end_atom_number = line.split()[1]
            data.append(end_atom_number)
        return data

    if mode == "residues_range":
        with open(file) as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith("2"):
                    data = "-".join([line.split()[3],line.split()[4]])
                    return data
                
def extract_coords_mod(file, atoms_data):
    with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
        with open(file) as f:
            atoms_data = [int(x) for x in atoms_data]
            total_atoms_solvated = str(atoms_data[-1])
            end_rec = atoms_data[0]
            rec_groups_number = len(atoms_data) -2
            start_pep = sum(atoms_data[:-2])+1
            end_pep = sum(atoms_data[:-2])+atoms_data[-2]
            if rec_groups_number >= 2:
                extra_rec_groups = atoms_data[1:-2]

            # ions_rec = []
            for line in f.readlines():
                line = line.strip("\n")
                if line.startswith("NTOTAL"):
                    line=" ".join(line.split(" ")[:-1]) + " " + total_atoms_solvated + '\n'
                    io.write(line)
                if line.startswith("LSTART"):
                    line=f'{" ".join(line.split(" ")[:-1])} {start_pep}\n'
                    io.write(line)
                if line.startswith("LSTOP"):
                    line=f'{" ".join(line.split(" ")[:-1])} {end_pep}\n'
                    io.write(line)
                if "NUMBER_REC_GROUPS" in line:
                    io.write(f"NUMBER_REC_GROUPS       {rec_groups_number}\n")
                if "RSTOP" in line:
                    line=f"RSTOP                   {end_rec}\n"
                    io.write(line)
                    if rec_groups_number >=2:
                        rec_atoms = end_rec
                        for rec in extra_rec_groups:
                            io.write(f"RSTART                  {rec_atoms + 1}\n")
                            rec_atoms += int(rec)
                            io.write(f"RSTOP                   {rec_atoms}\n")
                if not "NTOTAL" in line and not "LSTART" in line and not "LSTOP" in line and not "NUMBER_REC_GROUPS" in line and not "RSTOP" in line:
                    io.write(line + "\n")
    os.system(f"rm {file}")


def mod_in_file(residues_number, residues_number_prot):
    min_file = ["min.in"]
    for file in min_file:
        with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
            with open(file) as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if "restraintmask" in line:
                        line1 = line.split("restraintmask=")[0]
                        io.write(f"{line1}restraintmask=':1-{residues_number}!@H=',\n")
                    if "restraintmask" not in line:
                        io.write(line + "\n")

    min2_file = ["min2.in"]
    for file in min2_file:
        with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
            with open(file) as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if "restraintmask" in line:
                        line1 = line.split("restraintmask=")[0]
                        if cofactor_folder != None:
                            io.write(f"{line1}restraintmask=':1-{int(residues_number_prot)-1}!@H=',\n")
                        if cofactor_folder == None:
                            io.write(f"{line1}restraintmask=':1-{residues_number_prot}!@H=',\n")
                    if "restraintmask" not in line:
                        io.write(line + "\n")

    min3_file = ["min3.in"]
    for file in min3_file:
        with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
            with open(file) as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if "restraintmask" in line:
                        line1 = line.split("restraintmask=")[0]
                        if cofactor_folder != None:
                            io.write(f"{line1}restraintmask=':1-{int(residues_number_prot)-1}@CA,C,N,O',\n")
                        if cofactor_folder == None:
                            io.write(f"{line1}restraintmask=':1-{residues_number_prot}@CA,C,N,O',\n")
                    if "restraintmask" not in line:
                        io.write(line + "\n")
    os.system("rm min.in min2.in min3.in")

def mod_in_file_prod(residues_number, atoms_data):
    atoms_data = [int(x) for x in atoms_data]
    total_atoms_solvated = str(atoms_data[-1])
    end_rec = atoms_data[0]
    rec_groups_number = len(atoms_data) -2
    start_pep = sum(atoms_data[:-2])+1
    end_pep = sum(atoms_data[:-2])+atoms_data[-2]
    files = ["heat.in", "density.in"]
    for file in files:
        with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
            with open(file) as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if "restraintmask" in line:
                        line1 = line.split(":")[0]
                        io.write(f"{line1}:1-{residues_number}',\n")
                    if "restraintmask" not in line:
                        io.write(line + "\n")

    rmsd_file = "remove_water_prod_mdcrd.in"
    with open(f"{rmsd_file.split('.')[0]}_mod.{rmsd_file.split('.')[1]}", "w") as io:
        with open(rmsd_file) as f:
            for line in f.readlines():
                line = line.strip("\n")
                if "rms fit" in line:
                    line1 = line.split(":")[0]
                    io.write(f"{line1}:1-{residues_number}\n")
                if "rms fit" not in line:
                    io.write(line + "\n")

    rmsd_file = "measure_prod_rmsd.in"
    with open(f"{rmsd_file.split('.')[0]}_mod.{rmsd_file.split('.')[1]}", "w") as io:
        with open(rmsd_file) as f:
            for line in f.readlines():
                line = line.strip("\n")
                if "prod.rmsd" in line:
                    line1 = line.split("@")[0]
                    io.write(f"{line1}@{start_pep}-{end_pep}\n")
                if "prod.rmsd" not in line:
                    io.write(line + "\n")

    rmsd_file = "measure_prod_ligand_rmsf.in"
    with open(f"{rmsd_file.split('.')[0]}_mod.{rmsd_file.split('.')[1]}", "w") as io:
        with open(rmsd_file) as f:
            for line in f.readlines():
                line = line.strip("\n")
                if "atomicfluct" in line:
                    line1 = line.split("@")[0]
                    io.write(f"{line1}@{start_pep}-{end_pep}\n")
                if "atomicfluct" not in line:
                    io.write(line + "\n")
    os.system("rm heat.in density.in remove_water_prod_mdcrd.in measure_prod_rmsd.in measure_prod_ligand_rmsf.in")

def charge_check(file):

    chg_file = f'chg_{file}'
    os.system(f'obabel -isdf {file} -o sdf -O {chg_file} -p 7.0')
    supplier = Chem.SDMolSupplier(chg_file)
    os.system(f'rm {chg_file}')
    # Iterate over each molecule in the SDF file
    for mol in supplier:
        if mol is None:
            continue

        # Calculate formal charges
        AllChem.ComputeGasteigerCharges(mol)

        # Get the formal charges for each atom in the molecule
        formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]

        net_charge = 0
        for charge in formal_charges:
            charge = int(charge)
            net_charge += charge

        return net_charge

def mod_prod(new_nstlim):
    new_nstlim = int(int(new_nstlim) * 1000 / 0.002)
    total_frames = int(round(((new_nstlim * 5000) / 50000000), 0))
    new_ntpr = int(round((int(new_nstlim) / total_frames), 0))
    if new_ntpr == 0:
        new_ntpr += 1
    nfreq = int(round((total_frames / 100), 0))
    if nfreq == 0:
        nfreq += 1
    frames_evaluated = int(round((total_frames / nfreq), 0))
    with open('prod.in', 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if 'nstlim' in line:
            lines[i] = f"  nstlim={new_nstlim},dt=0.002,\n"
        elif 'ntpr' in line:
            lines[i] = f"  ntpr={new_ntpr}, ntwx={new_ntpr},\n"
    with open('prod_mod.in', 'w') as file:
        file.writelines(lines)

    os.system(f'mv extract_coords_prod_mod.mmpbsa extract_coords_prod.mmpbsa') 
    with open('extract_coords_prod.mmpbsa', 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if 'NSTOP' in line:
            lines[i] = f"NSTOP                   {total_frames}\n"
        if 'NFREQ' in line:
            lines[i] = f"NFREQ                   {nfreq}\n"
    with open('extract_coords_prod_mod.mmpbsa', 'w') as file:
        file.writelines(lines)

    with open('binding_energy_prod.mmpbsa', 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if 'PARALLEL' in line:
            lines[i] = f"PARALLEL              {threads}\n"
        if 'STOP' in line:
            lines[i] = f"STOP                  {frames_evaluated}\n"
    with open('binding_energy_prod_mod.mmpbsa', 'w') as file:
        file.writelines(lines)
    
    os.system("rm prod.in binding_energy_prod.mmpbsa extract_coords_prod.mmpbsa")

def prepare_ligand(ligand_name):
    cp_in_files = f"cp {in_folder}/lig_or_pep/leap_ligand.in ."
    os.system(cp_in_files)
    cp_sdf = f"mv ../{ligand_name} ."
    os.system(cp_sdf)
    os.system(f"mv {ligand_name} lig.sdf")
    charge_ligand = charge_check("lig.sdf")
    print(f'---------------------------------------\nPreparing ligand...')
    print(f"Net charge of ligand is {charge_ligand}")
    cmd_mol2_to_prepi = f"antechamber -i lig.sdf -fi sdf -o lig.mol2 -fo mol2 -dr n -at gaff2 -c bcc -nc {charge_ligand} -pf Y> antechamber_ligand.log"
    os.system(cmd_mol2_to_prepi)
    cmd_frcmod = "parmchk2 -i lig.mol2 -o lig.frcmod -f mol2 -a Y"
    os.system(cmd_frcmod)
    cmd_leap_ligand = "tleap -s -f leap_ligand.in > leap_ligand.out"
    os.system(cmd_leap_ligand)
    print(f'Done')

def prepare_peptide(peptide_name):
    format = peptide_name.split(".")[-1]
    peptide_name = peptide_name.split(".")[0]
    os.system(f"cp {in_folder}/lig_or_pep/leap_ligand.in .")
    os.system(f"mv ../{peptide_name}.{format} .")
    os.system(f'mv {peptide_name}.{format} og_lig.pdb')
    os.system(f"pdb4amber -i og_lig.pdb -y --add-missing-atoms --reduce -d -p -o lig.pdb")
    os.system(f'obabel -ipdb lig.pdb -o sdf -O lig.sdf')
    charge_ligand = charge_check("lig.sdf")
    print(f'---------------------------------------\nPreparing peptide...')
    print(f"Net charge of ligand is {charge_ligand}")
    cmd_mol2_to_prepi = f"antechamber -i lig.sdf -fi sdf -o lig.mol2 -fo mol2 -dr n -at gaff2 -c bcc -nc {charge_ligand} -pf Y> antechamber_peptide.log"
    os.system(cmd_mol2_to_prepi)
    cmd_frcmod = "parmchk2 -i lig.mol2 -o lig.frcmod -f mol2 -a Y"
    os.system(cmd_frcmod)
    cmd_leap_ligand = "tleap -s -f leap_ligand.in > leap_peptide.out"
    os.system(cmd_leap_ligand)
    print("Done")

def prepare_receptor(receptor_file):
    receptor_name = receptor_file.split('/')[-1].replace(".pdb", "")
    receptor_folder = f"rec_{receptor_name}"
    if not os.path.exists(receptor_folder):
        os.mkdir(receptor_folder)
    os.chdir(receptor_folder)
    os.system(f"cp {in_folder}/core/* .")
    os.system(f"cp {in_folder}/lig_or_pep/*ligand* .")
    os.system(f"cp {receptor_file} .")
    os.system(f"mv {receptor_name}.pdb og_receptor.pdb")
    os.system(f"pdb4amber -i og_receptor.pdb -y --add-missing-atoms --reduce -d -p -o receptor_mod.pdb")
    if cofactor_folder != None:
        os.system(f"cp {in_folder}/cofactor/leap_cofactor.in .")
        if os.path.isdir(cofactor_folder):
            os.system(f"cp {cofactor_folder}/{prefix_cofactor}{receptor_name}.sdf .")
            os.system(f"mv {prefix_cofactor}{receptor_name}.sdf cofactor.sdf")
        else:
            cofactor_file = cofactor_folder
            os.system(f'cp {cofactor_file} .')
            os.system(f'mv {cofactor_file.split("/")[-1]} cofactor.sdf')
        charge_cofactor = charge_check("cofactor.sdf")
        print(f'---------------------------------------\nPreparing cofactor...')
        print(f"Net charge of cofactor is {charge_cofactor}")
        cmd_mol2_to_prepi = f"antechamber -i cofactor.sdf -fi sdf -o cofactor.mol2 -fo mol2 -at gaff2 -c bcc -nc {charge_cofactor} -pf Y> antechamber_cofactor.log"
        os.system(cmd_mol2_to_prepi)
        cmd_frcmod = "parmchk2 -i cofactor.mol2 -o cofactor.frcmod -f mol2 -a Y"
        os.system(cmd_frcmod)
        cmd_leap_ligand = "tleap -s -f leap_cofactor.in > leap_cofactor.out"
        os.system(cmd_leap_ligand)
        print('Done')


def run_dynamics(session_dir, receptor_file, ligand_name, gpu_num):
    os.chdir(session_dir)
    ligand_folder = f'run_{ligand_name.replace(".sdf", "").replace(".pdb", "")}'
    os.chdir(ligand_folder)
    receptor_name = receptor_file.split('/')[-1].replace(".pdb", "")
    receptor_folder = f"rec_{receptor_name}"
    os.system(f"cp -r ../{receptor_folder} .")
    os.chdir(receptor_folder)
    os.system(f"cp ../lig.* .")
    # if ligand_type:
    if cofactor_folder != None:
        os.system(f"cp {in_folder}/cofactor/*ligand* .")
        print(f'---------------------------------------\nPreparing complex with cofactor...')
        cmd_leap = "tleap -s -f leap_commands_ligand_cofactor_prot.in > leap_lig_cofactor_prot.out"
        os.system(cmd_leap)
        print('Done')
    if cofactor_folder == None:
        print(f'---------------------------------------\nPreparing complex...')
        cmd_leap = "tleap -s -f leap_commands_ligand_prot.in > leap_lig_prot.out"
        os.system(cmd_leap)
        print('Done')
    parminfo_complex = "cpptraj complex-no_water.prmtop -i parminfo.in > parminfo_complex.out"
    os.system(parminfo_complex)
    molinfo = "cpptraj complex-no_water.prmtop -i molinfo.in > molinfo_complex.out"
    os.system(molinfo)
    residues_number_prot = data_seeker("molinfo_complex.out", "residues", "residues_number_prot")
    residues_number = data_seeker("parminfo_complex.out", "residues", "residues_number")
    mod_in_file(residues_number, residues_number_prot)
    parminfo = "cpptraj complex_solvated.prmtop -i parminfo.in > parminfo_solvated.out"
    os.system(parminfo)
    atoms_number = data_seeker("parminfo_solvated.out", "Topology", "atoms_number")
    atom_extract_info = "cpptraj complex-no_water.prmtop -i molinfo.in > molinfo_no_water.out"
    os.system(atom_extract_info)
    atoms_data = data_seeker("molinfo_no_water.out", "None", "atoms_data")
    atoms_data.append(atoms_number)
    extract_coords_mod("extract_coords.mmpbsa", atoms_data)
    if not os.path.exists("coords"):
        os.mkdir("coords")
    if not os.path.exists("FAR_results"):
        os.mkdir("FAR_results")
    print(f'---------------------------------------\nRunning minimizations...')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    tmux_send_run = f'python3 run_commands.py FAR_commands.txt'
    os.system(tmux_send_run)
    os.system("cpptraj -i get_min_complex.in")
    convert_trj3 = f'cpptraj -p complex_solvated.prmtop -y min3.rst -x min3.trj'
    os.system(convert_trj3)
    extrac_coords_cmd = f'$AMBERHOME/bin/mm_pbsa.pl extract_coords_mod.mmpbsa 1> extract_coords.log 2>extract_coords.err'
    os.system(extrac_coords_cmd)
    print('Done')
    print(f'---------------------------------------\nCalculating binding affinity')
    bindingE_cmd = f'$AMBERHOME/bin/mm_pbsa.pl binding_energy.mmpbsa > binding_energy.log'
    os.system(bindingE_cmd)
    print('Done')
    os.system("mv min_complex.pdb FAR_results")
    os.system(f'mv snapshot* FAR_results')
    os.system(f'mv coords extract_coords.err extract_coords.log binding_energy.log FAR_results; mkdir coords')
    #Here starts prod commands
    if prod:
        os.system(f"cp {in_folder}/prod/* .")
        extract_coords_mod("extract_coords_prod.mmpbsa", atoms_data)
        mod_in_file_prod(residues_number, atoms_data)
        mod_prod(time_prod)
        print(f'---------------------------------------\nRunning molecular dynamics...')
        tmux_send_run2 = f'python3 run_commands.py commands.txt'
        os.system(tmux_send_run2)
        convert_trj = f'cpptraj -p complex_solvated.prmtop -y prod.mdcrd -x prod.trj'
        os.system(convert_trj)
        remove_wat_cmd = f'cpptraj -i remove_water_prod_mdcrd_mod.in'
        os.system(remove_wat_cmd)
        os.system(f'cpptraj -i measure_prod_rmsd_mod.in')
        os.system(f'cpptraj -i measure_prod_ligand_rmsf_mod.in')
        extrac_coords_cmd = f'$AMBERHOME/bin/mm_pbsa.pl extract_coords_prod_mod.mmpbsa 1> extract_coords_prod.log 2>extract_coords_prod.err'
        os.system(extrac_coords_cmd)
        bindingE_cmd = f'$AMBERHOME/bin/mm_pbsa.pl binding_energy_prod_mod.mmpbsa > binding_energy_prod.log'
        os.system(bindingE_cmd)
    print("FAR protocol finished")

def extract_SDF(ligand):
    ligand_name_list = []
    with open(ligand) as io:
        line_list = io.readlines()
        indexes = []
        for idx, line in enumerate(line_list):
            if '$$$$' in line:
                indexes.append(idx)

        for i in range(len(indexes)):
            start = 0 if i == 0 else indexes[i - 1] + 1
            end = indexes[i]
            output_file = f'{line_list[start].replace("|", "_").strip().replace(".pdb", "")}.sdf'
            with open(output_file, "w") as of:
                data_list = line_list[start:end + 1]
                for line in data_list:
                    of.write(line)
                ligand_name_list.append(output_file)
    return ligand_name_list

def table_generator():
    energy_patch_list = []
    with open("failed_files.txt", "w") as outf:
        outf.write("ligand_name\treceptor_name\n")
        for ligand_folder in next(os.walk("."))[1]:
            for receptor_folder in next(os.walk(ligand_folder))[1]:
                ligand_name = ligand_folder.replace("run_", "")
                receptor_name = receptor_folder.replace("rec_", "")
                try:
                    file = f'{ligand_folder}/{receptor_folder}/FAR_results/snapshot_statistics.out'
                    with open(file) as io:
                        line = io.readlines()[-1].strip()
                        energy_value = line.split()[1]
                        energy_patch = float(energy_value), ligand_name, receptor_name
                        energy_patch_list.append(energy_patch)
                except:
                    outf.write(f'{ligand_name}\t{receptor_name}\n')
                    continue

    sorted_list = sorted(energy_patch_list,key=itemgetter(1))
    if len(sorted_list) >= 1:
        with open("FAR_results.tsv", "w") as outfile:
            outfile.write('AffinityBindingPred(kcal/mol)\tLigand\tReceptor\n')
            for binding_data in sorted_list:
                energy_value = binding_data[0]
                ligand_name = binding_data[1]
                receptor_name = binding_data[2]
                output = f'{energy_value}\t{ligand_name}\t{receptor_name}\n'
                outfile.write(output)

if __name__ == '__main__':
    gpu_ids = list(range(range_GPU_S,range_GPU_E))
    if not prod_only:
        initial_path = os.getcwd()
        ligand_name = ligand.split("/")[-1]
        if cofactor_folder == None:
            session_name = f'{initial_path}/run_{ligand_name.replace(".sdf", "").replace(".pdb", "")}'
        if cofactor_folder != None:
            session_name = f'{initial_path}/run_{ligand_name.replace(".sdf", "").replace(".pdb", "")}_wCofactor'
        if not os.path.exists(session_name):
            os.mkdir(session_name)
        os.chdir(session_name)
        if os.path.isdir(ligand):
            ligand_name_list = []
            for file in os.listdir(ligand):
                format = file.split(".")[-1]
                if format =="pdb":
                    os.system(f'cp {ligand}/{file} .')
                    ligand_name_list.append(file)
                if format == "sdf":
                    ligand_name_list = extract_SDF(file)
                    os.system(f"rm {file}")
            session_dir = os.getcwd()
            
        else:
            os.system(f'cp {ligand} .')
            ligand = ligand.split("/")[-1]
            format = ligand.split(".")[-1]
            if format == "sdf":
                ligand_name_list = extract_SDF(ligand)
                os.system(f"rm {ligand}")
            if format =="pdb":
                ligand_name_list = [ligand]
            session_dir = os.getcwd()

        def ligand_functions(ligand_name):
            os.chdir(session_dir)
            ligand_folder = f'run_{ligand_name.replace(".sdf", "").replace(".pdb", "")}'
            if not os.path.exists(ligand_folder):
                os.mkdir(ligand_folder)
            os.chdir(ligand_folder)
            if ligand_type:
                prepare_ligand(ligand_name)
            if peptide_type:
                prepare_peptide(ligand_name)

        with multiprocessing.Pool(processes=threads) as pool:
            pool.map(ligand_functions, ligand_name_list)

        if os.path.isdir(receptor):
            receptor_list = os.listdir(receptor)
            receptor_list = [f'{receptor}/{receptor_file}' for receptor_file in receptor_list]
            with multiprocessing.Pool(processes=threads) as pool:
                pool.map(prepare_receptor, receptor_list)
            if len(receptor_list) >= len(ligand_name_list):
                gpu_cycle = itertools.cycle(gpu_ids)
                with multiprocessing.Pool(processes=threads) as pool:
                    args = [(session_dir, receptor_file, ligand_name, next(gpu_cycle)) for ligand_name in ligand_name_list for receptor_file in receptor_list]
                    pool.starmap(run_dynamics, args)
            if len(receptor_list) < len(ligand_name_list):
                gpu_cycle = itertools.cycle(gpu_ids)
                with multiprocessing.Pool(processes=threads) as pool:
                    args = [(session_dir, receptor_file, ligand_name, next(gpu_cycle)) for receptor_file in receptor_list for ligand_name in ligand_name_list]
                    pool.starmap(run_dynamics, args)
        else:
            os.system(f'cp {receptor} {session_name}')
            receptor_list = [receptor.split("/")[-1]]
            receptor_list = [f'{session_name}/{receptor_file}' for receptor_file in receptor_list]
            with multiprocessing.Pool(processes=threads) as pool:
                pool.map(prepare_receptor, receptor_list)
            gpu_cycle = itertools.cycle(gpu_ids)
            with multiprocessing.Pool(processes=threads) as pool:
                args = [(session_dir, receptor_list[0], ligand_name, next(gpu_cycle)) for ligand_name in ligand_name_list]
                pool.starmap(run_dynamics, args)
            os.system(f"rm {receptor_list[0]}")

        os.chdir(session_name)
        os.system("rm -r rec*")
        table_generator()

    if prod_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(range_GPU_S)
        atoms_number = data_seeker("parminfo_solvated.out", "Topology", "atoms_number")
        residues_number = data_seeker("parminfo_complex.out", "residues", "residues_number")
        atoms_data = data_seeker("molinfo_no_water.out", "None", "atoms_data")
        atoms_data.append(atoms_number)
        os.system(f"cp {in_folder}/prod/* .")
        extract_coords_mod("extract_coords_prod.mmpbsa", atoms_data)
        mod_in_file_prod(residues_number, atoms_data)
        mod_prod(time_prod)
        print(f'---------------------------------------\nRunning molecular dynamics...')
        tmux_send_run2 = f'python3 run_commands.py commands.txt'
        os.system(tmux_send_run2)
        convert_trj = f'cpptraj -p complex_solvated.prmtop -y prod.mdcrd -x prod.trj'
        os.system(convert_trj)
        remove_wat_cmd = f'cpptraj -i remove_water_prod_mdcrd_mod.in'
        os.system(remove_wat_cmd)
        os.system(f'cpptraj -i measure_prod_rmsd_mod.in')
        os.system(f'cpptraj -i measure_prod_ligand_rmsf_mod.in')
        extrac_coords_cmd = f'$AMBERHOME/bin/mm_pbsa.pl extract_coords_prod_mod.mmpbsa 1> extract_coords_prod.log 2>extract_coords_prod.err'
        os.system(extrac_coords_cmd)
        bindingE_cmd = f'$AMBERHOME/bin/mm_pbsa.pl binding_energy_prod_mod.mmpbsa > binding_energy_prod.log'
        os.system(bindingE_cmd)
        print("FAR protocol finished")