def modify_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    modified_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            if float(parts[-3]) > 1:
                parts[-3] = str(0.9)
        modified_lines.append(' '.join(parts) + '\n')

    with open(output_file, 'w') as outfile:
        outfile.writelines(modified_lines)

input_file = 'centauro_pick.txt'
output_file = 'centauro_pick_modifid.txt'
modify_file(input_file, output_file)
