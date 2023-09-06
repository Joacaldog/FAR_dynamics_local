import os
from multiprocessing.pool import Pool
import re
import argparse

program_description = "Runs FAR protocol"
parser = argparse.ArgumentParser(description=program_description)
parser.add_argument("-l", "--ligand", type=str,
                    help="Ligand file in SDF format", required=True)
parser.add_argument("-r", "--receptor", type=str,
                    help="Receptor file in PDB format", required=True)
parser.add_argument("-GPU", "--GPU_range", type=str,
                    help="Specify GPUs from 0 to 7 (e.g: 0-3 to utilize GPUs 0,1,2)", required=True)
parser.add_argument("-in", "--in_folder", type=str,
                    help="Specify folder absolute path where all '.in' files are located (e.g: /home/user/FAR_and_prod_in_files)", required=True)
parser.add_argument("-P", "--prod", action="store_true",
                    help="Optional: Runs 100ns of molecular dynamics production phase and utilizes MMPBSA for calculation of binding free energy")
parser.add_argument("-co", "--cofactor", nargs='?',
                    help="Optional: cofactor file in SDF format")


args = parser.parse_args()
ligand = args.ligand
receptor = args.receptor
complex_file = f'{ligand.replace(".sdf", "")}'
range_GPU = args.GPU_range
in_folder = args.in_folder
range_GPU_S = int(range_GPU.split("-")[0])
range_GPU_E = int(range_GPU.split("-")[1])
cofactor = args.cofactor
prod = args.prod

if prod != True:
    print("Running FAR protocol")

if prod == True:
    print("Running FAR protocol and 100ns of molecular dynamics production with MMPBSA calculation")

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

                

def mod_in_file(residues_number):
    files = ["min2.in"]
    for file in files:
        with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
            with open(file) as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if "restraintmask" in line:
                        line = line.split(":")[0]
                        io.write(f"{line}:1-{residues_number}!@H=',\n")
                    if "restraintmask" not in line:
                        io.write(line + "\n")

def mod_in_file2(residues_number):
    files = ["heat.in", "density.in"]
    for file in files:
        with open(f"{file.split('.')[0]}_mod.{file.split('.')[1]}", "w") as io:
            with open(file) as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    if "restraintmask" in line:
                        line = line.split(":")[0]
                        io.write(f"{line}:1-{residues_number}',\n")
                    if "restraintmask" not in line:
                        io.write(line + "\n")

    rmsd_file = "remove_water_prod_mdcrd.in"
    with open(f"{rmsd_file.split('.')[0]}_mod.{rmsd_file.split('.')[1]}", "w") as io:
        with open(rmsd_file) as f:
            for line in f.readlines():
                line = line.strip("\n")
                if "rms fit" in line:
                    line = line.split(":")[0]
                    io.write(f"{line}:1-{residues_number}\n")
                if "rms fit" not in line:
                    io.write(line + "\n")
    os.system("rm heat.in density.in remove_water_prod_mdcrd.in")

def charge_check(file):
    final_charge = 0
    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            if ("CHG") in line:
                charge = line.split()[-1]
                if charge != "END":
                    charge = int(charge)
                    final_charge += charge
    return final_charge

def run_dynamics(i):
    gpu_num = i
    rep_folder = f"{complex_file}_run_{i}"
    if not os.path.exists(rep_folder):
        os.mkdir(rep_folder)
    os.chdir(rep_folder)
    cp_in_files = f"cp {in_folder}/* ."
    os.system(cp_in_files)
    cp_sdf = f"cp ../{ligand} ."
    os.system(cp_sdf)
    os.system(f"mv {ligand} pep.sdf")
    cp_pdb3 = f"cp ../{receptor} ."
    os.system(cp_pdb3)
    os.system(f"mv {receptor} og_receptor.pdb")
    os.system(f"pdb4amber -i og_receptor.pdb -y --reduce --add-missing-atoms -o receptor_mod.pdb")
    charge_ligand = charge_check("pep.sdf")
    print(f'---------------------------------------\nPreparing ligand...')
    print(f"Net charge of ligand is {charge_ligand}")
    cmd_mol2_to_prepi = f"antechamber -i pep.sdf -fi sdf -o pep.mol2 -fo mol2 -c bcc -nc {charge_ligand} -pf Y> antechamber_ligand.log"
    os.system(cmd_mol2_to_prepi)
    cmd_frcmod = "parmchk2 -i pep.mol2 -o pep.frcmod -f mol2 -a Y"
    os.system(cmd_frcmod)
    cmd_leap_ligand = "tleap -s -f leap_ligand.in > leap_ligand.out"
    os.system(cmd_leap_ligand)
    print(f'Done')
    if cofactor != None:
        os.system(f"cp ../{cofactor} .")
        os.system(f"mv {cofactor} cofactor.sdf")
        charge_cofactor = charge_check("cofactor.sdf")
        print(f'---------------------------------------\nPreparing cofactor...')
        print(f"Net charge of cofactor is {charge_cofactor}")
        cmd_mol2_to_prepi = f"antechamber -i cofactor.sdf -fi sdf -o cofactor.mol2 -fo mol2 -c bcc -nc {charge_cofactor} -pf Y> antechamber_cofactor.log"
        os.system(cmd_mol2_to_prepi)
        cmd_frcmod = "parmchk2 -i cofactor.mol2 -o cofactor.frcmod -f mol2 -a Y"
        os.system(cmd_frcmod)
        cmd_leap_ligand = "tleap -s -f leap_cofactor.in > leap_cofactor.out"
        os.system(cmd_leap_ligand)
        print('Done')
        print(f'---------------------------------------\nPreparing complex with cofactor...')
        cmd_leap = "tleap -s -f leap_commands_ligand_cofactor_prot.in > leap_lig_cofactor_prot.out"
        os.system(cmd_leap)
        print('Done')
    if cofactor == None:
        print(f'---------------------------------------\nPreparing complex...')
        cmd_leap = "tleap -s -f leap_commands_ligand_prot.in > leap_lig_prot.out"
        os.system(cmd_leap)
        print('Done')
    parminfo_complex = "cpptraj complex-no_water.prmtop -i parminfo.in > parminfo_complex.out"
    os.system(parminfo_complex)
    molinfo = "cpptraj complex-no_water.prmtop -i molinfo.in > molinfo_complex.out"
    os.system(molinfo)
    residues_number_prot = data_seeker("molinfo_complex.out", "residues", "residues_number_prot")
    mod_in_file(residues_number_prot)
    residues_number = data_seeker("parminfo_complex.out", "residues", "residues_number")
    mod_in_file2(residues_number)
    parminfo = "cpptraj complex_solvated.prmtop -i parminfo.in > parminfo_solvated.out"
    os.system(parminfo)
    atoms_number = data_seeker("parminfo_solvated.out", "Topology", "atoms_number")
    atom_extract_info = "cpptraj complex-no_water.prmtop -i molinfo.in > molinfo_no_water.out"
    os.system(atom_extract_info)
    atoms_data = data_seeker("molinfo_no_water.out", "None", "atoms_data")
    atoms_data.append(atoms_number)
    extract_coords_mod("extract_coords.mmpbsa", atoms_data)
    extract_coords_mod("extract_coords_prod.mmpbsa", atoms_data)
    if not os.path.exists("coords"):
        os.mkdir("coords")
    if not os.path.exists("FAR_results"):
        os.mkdir("FAR_results")
    print(f'---------------------------------------\nRunning minimizations...')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    tmux_send_run = f'python3 run_commands.py FAR_commands.txt'
    os.system(tmux_send_run)
    convert_trj3 = f'cpptraj -p complex_solvated.prmtop -y min3.rst -x min3.trj'
    os.system(convert_trj3)
    extrac_coords_cmd = f'$AMBERHOME/bin/mm_pbsa.pl extract_coords_mod.mmpbsa 1> extract_coords.log 2>extract_coords.err'
    os.system(extrac_coords_cmd)
    print('Done')
    print(f'---------------------------------------\nCalculating binding affinity')
    bindingE_cmd = f'$AMBERHOME/bin/mm_pbsa.pl binding_energy.mmpbsa > binding_energy.log'
    os.system(bindingE_cmd)
    print('Done')
    os.system(f'mv snapshot* FAR_results')
    os.system(f'mv coords extract_coords.err extract_coords.log binding_energy.log FAR_results; mkdir coords')
    #Here starts prod commands
    if prod == True:
        print(f'---------------------------------------\nRunning molecular dynamics...')
        tmux_send_run2 = f'python3 run_commands.py commands.txt'
        os.system(tmux_send_run2)
        convert_trj = f'cpptraj -p complex_solvated.prmtop -y prod.mdcrd -x prod.trj'
        os.system(convert_trj)
        remove_wat_cmd = f'cpptraj -i remove_water_prod_mdcrd_mod.in'
        os.system(remove_wat_cmd)
        os.system(f'cpptraj -i measure_prod_rmsd.in')
        os.system(f'cpptraj -i measure_prod_rmsf.in')
        extrac_coords_cmd = f'$AMBERHOME/bin/mm_pbsa.pl extract_coords_prod_mod.mmpbsa 1> extract_coords_prod.log 2>extract_coords_prod.err'
        os.system(extrac_coords_cmd)
        bindingE_cmd = f'$AMBERHOME/bin/mm_pbsa.pl binding_energy.mmpbsa > binding_energy_prod.log'
        os.system(bindingE_cmd)
    print("FAR protocol finished")
        
# protect the entry point
if __name__ == '__main__':
    # create and configure the process pool
    with Pool() as pool:
        # execute tasks, block until all completed
        pool.map(run_dynamics, range(range_GPU_S,range_GPU_E))
    # process pool is closed automatically