import os
path = os.getcwd()
fs = os.listdir(path)

for d in fs:
	if d[:4] == "test":
		new_name = "2015" + d[4:]
		os.rename(os.path.join(path, d), os.path.join(path, new_name))

