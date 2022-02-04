
import os

path = "/mnt/data/barrett/4classtraining/data/"

lijst = set()
for folder in os.listdir(path):
    # lijst.add(folder)
    lijst.add(folder[:-2])

# lijst.sort
# print(lijst)

lijst = list(lijst)
lijst.sort()

if __name__ == '__main__':

    for i, filename in enumerate(lijst):
        prefix = str(i % 10)
        # print(f'prefix:{prefix}')
        for j in range(1, 5):
            if os.path.isdir(os.path.join(path, filename + "-" + str(j))):
                l = os.listdir(os.path.join(path, filename + "-" + str(j)))
                for file in l:
                    os.rename(os.path.join(path, filename + "-" + str(j), file), os.path.join(path, prefix + file))

    for file in lijst:
        if file.endswith("png"):
            os.rename(os.path.join(path, file), os.path.join(path, "data", file))

    print("done")


