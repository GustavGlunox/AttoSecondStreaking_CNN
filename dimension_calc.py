import numpy as np

array = [800,40] # [H,W]

# Rechnung Definieren
def outdimension(array,stride=1,*args):
    try:
        new_array=[]
        for a,b in zip(array,args):
            number = (a-b)/stride
            new = np.floor(number)+1
            new_array.append(new)
        return new_array
    except Exception as e:
        print(f"Länge von *args muss {len(array)} sein!")
        print(e)

# Dimensionen berechnen

# Block 1.1
array=outdimension(array,1,21,8)
print(array)

# Block 1.2
array=outdimension(array,3,13,5)
print(array)

# Block 2.1
array=outdimension(array,1,13,5)
print(array)

# Block 2.2
array=outdimension(array,2,9,3)
print(array)

# Block 3.1
array=outdimension(array,1,3,2)
print(array)

# Block 3.2
array=outdimension(array,2,2,1)
print(array)

# Dimension output:
array.append(40)
array = np.array(array)
dim = np.prod(array)
print(dim)
