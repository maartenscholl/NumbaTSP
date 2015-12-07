# 

def parse(filename):
    '''
    Opportunistically parse a tsplib problem set
    '''
    with open(filename) as file:
        lines = list(map(str.strip, filter(None, file.read().splitlines())))
        vertices = dict()
        labels = dict()
        start = lines.index('NODE_COORD_SECTION')
        count = 0
        for item in lines[start:]:
            if item[0].isdigit():
                index, first_coord, second_coord  = item.split()
                vertices[count] = (float(first_coord), float(second_coord))
                labels[count] = int(index)
                count += 1
        return vertices, labels