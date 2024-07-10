import json
import matplotlib.pyplot as plt
import numpy as np

# # Sample list of droplet areas (you can replace this with your actual data)
# droplet_areas = [2, 3, 5, 6, 2, 1, 1, 4, 5, 6, 4, 5, 3, 2, 3, 5, 4, 5, 6, 7]
def create_graph_values(droplet_area):
    droplet_area = [2, 3, 5, 6, 2, 1, 1, 4, 5, 6, 4, 5, 3, 2, 3, 5, 4, 5, 6, 7, 16, 10, 13, 19]

    area_ranges = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21),]  # Example ranges, adjust as needed

    # Initialize counts for each range
    counts = [0] * len(area_ranges)

    # Count droplets in each range
    for area in droplet_area:
        for i, (low, high) in enumerate(area_ranges):
            if low <= area <= high:
                counts[i] += 1
                break 

    # Plotting the bar chart
    # ranges_str = [f"{low}-{high}" for (low, high) in area_ranges]
    # plt.bar(ranges_str, counts, color='skyblue')
    # plt.xlabel('Area Ranges')
    # plt.ylabel('Number of Droplets')
    # plt.title('Droplet Areas Distribution')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()



    # # Write to JSON file
    # json_filename = 'droplet_areas.json'
    # with open(json_filename, 'w') as json_file:
    #     json.dump(data, json_file, indent=4)

    return counts