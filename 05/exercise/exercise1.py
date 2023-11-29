from pathlib import Path

if __name__ == "__main__":
    
    data_dir = "./data"
    data_dir_path = Path(data_dir).resolve()
    print("---Displaying the absolute path---")
    print(data_dir_path)

    dir_list = list(data_dir_path.glob("*"))
    print("---Displaying all files underneath")
    for dir in dir_list:
        print(dir)

    file_list = []
    for dir in dir_list:
        file_path_list = list(dir.glob("*"))
        file_list += file_path_list
    print("===== problem3 =====")
    print(len(file_list))