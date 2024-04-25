def save_circle_info(circles):
    circle_list = []

    # Sort circles by their x-coordinate to make sure the IDs are assigned correctly
    circles = sorted(circles, key=lambda x: x[0])

    # Generate IDs for circles
    circle_ids = list(range(1, len(circles) + 1))

    for i, circle in enumerate(circles):
        circle_info = {
            "id": circle_ids[i],
            "center": (circle[0], circle[1]),
            "radius": circle[2],
            "overlapped_ids": circle_ids[:i] + circle_ids[i+1:]
        }
        circle_list.append(circle_info)

    return circle_list

# Example usage:
detected_circles = [(100, 100, 20), (150, 150, 25), (200, 200, 30), (250, 250, 35)]
circle_info_list = save_circle_info(detected_circles)

for circle_info in circle_info_list:
    print("Circle ID:", circle_info["id"])
    print("Center:", circle_info["center"])
    print("Radius:", circle_info["radius"])
    print("Overlapped IDs:", circle_info["overlapped_ids"])
    print()
