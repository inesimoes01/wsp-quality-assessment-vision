class GroundTruth_Statistics:
    def __init__(self, stats_file):
        # read file
        with open(stats_file, 'r') as f:
            lines = f.readlines()
            self.no_total_droplets = int(lines[0].split(":")[1].strip())
            self.coverage_percentage = float(lines[1].split(":")[1].strip())
            self.vmd_value = float(lines[2].split(":")[1].strip())
            self.rsf_value = float(lines[3].split(":")[1].strip())
            self.no_overlapped_droplets = int(lines[4].split(":")[1].strip())

            # Find the index where the overlapped droplets section begins
            self.overlapped_droplets = {}

            # Iterate over the lines containing information about overlapped droplets
            for line in lines[7:]:  # Start from line 7 to skip the header lines
                # Split the line to extract droplet number and its overlapping droplets
                parts = line.strip().split(': ')
                droplet_number = int(parts[0].split()[-1])
                overlapping_droplets = [int(d) for d in parts[1][1:-1].split(', ')]
                
                # Store the overlapped droplets in the dictionary
                self.overlapped_droplets[droplet_number] = overlapping_droplets 

            # for line in f:
            #     if "Number of droplets: " in line:
            #         self.no_total_droplets = int(line.split(":")[1].strip())
            #     if "Coverage percentage: " in line:
            #         self.coverage_percentage = int(line.split(":")[1].strip())
            #     if "VMD value: " in line:
            #         self.vmd_value = int(line.split(":")[1].strip())
            #     if "RSF value: " in line:
            #         self.coverage_percentage = int(line.split(":")[1].strip())
            #     if "Number of overlapped droplets: " in line:
            #         self.no_overlapped_droplets = int(line.split(":")[1].strip())
            #     if "OVERLAPPED DROPLETS" in line:
 
                    
            #         parts = line.split(':')
            #         droplet_no = int(parts[0].split()[-1])  # Extract droplet number
            #         overlapped_droplets = [int(x) for x in parts[1].strip()[1:-1].split(',')]  # Extract overlapped droplets
            #         # Store the information in the dictionary
            #         drop[droplet_no] = overlapped_droplets
            #         self.coverage_percentage = int(line.split(":")[1].strip())

    #     self.get_overlapping()
    #     self.get_VMD()
    #     self.get_VMD()

    # def get_overlapping(self):
    #     return

    # def get_VMD(self):
    #     return

    # def get_VMD(self):
    #     return