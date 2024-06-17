import requests

def get_data_from_server():
    url = 'http://127.0.0.1:5000/data'  # URL of the Flask server endpoint
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()  # Parse the JSON response
        return data
    else:
        return None

if __name__ == '__main__':
    data = get_data_from_server()
    if data:
        print(f"Name: {data['name']}")
        print(f"Age: {data['age']}")
        print(f"City: {data['city']}")
    else:
        print("Failed to retrieve data")
