from movenet import MoveNet
import cv2

data_dir = 'K:/Data/SCData'

mvnet = MoveNet()

pos_lst = []
labels = []
img_lst = []

with open(f'{data_dir}/index.txt', 'r') as f:
    for line in f:
        line = line.split()
        img_lst.append(f'{data_dir}/{line[0]}')
        labels.append(int(line[1]))

for fp in img_lst:
    print(fp)
    img = cv2.imread(fp)
    keypoints = mvnet.run(img).flatten(order='C')
    keypoints = keypoints[[0, 1, 3, 4, 6, 7, 9, 10, 12,
                           13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31]]
    pos_lst.append(keypoints)

with open('./tmp/pos.txt', 'w') as f:
    for i in range(len(labels)):
        f.write(str(labels[i]))
        for j in pos_lst[i]:
            f.write(f' {j}')
        f.write('\n')

fin = open('./tmp/pos.txt', 'r')
fout = open('./tmp/pos.csv', 'w')

for line in fin:
    line = line.split()
    fout.write(",".join(line))
    fout.write('\n')

fin.close()
fout.close()
