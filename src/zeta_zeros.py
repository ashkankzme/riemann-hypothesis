def read_txt_file_line_by_line(filepath: str) -> list[str]:
    with open(filepath, 'r') as file:
        # strip the trailing spaces and newlines
        return [line.strip() for line in file.readlines()]
