
import csv


hashname = {}
with open('MalwareBazaar_Labels.csv','r') as f:
	for line in f:
		hashname.setdefault(line.split(',')[0],line.split(',')[1].strip())



thefilepath = 'degree_features.csv'
count = -1
for count, line in enumerate(open(thefilepath, 'r')):
	pass
count += 1
print(count)
feature_csv = [[] for i in range(count)]

tmp = []
with open(thefilepath, "r") as f:
	csv_reader = csv.reader(f)
	list_of_rows = list(csv_reader)
	i = 0
	for row in list_of_rows:
		if i == 0:
			i = i + 1
		else:
			if hashname[row[0]] == 'Gozi':
				row[-1] = '1'
			elif hashname[row[0]] == 'GuLoader':
				row[-1] = '2'
			elif hashname[row[0]] == 'Heodo':
				row[-1] = '3'
			elif hashname[row[0]] == 'IcedID':
				row[-1] = '4'
			elif hashname[row[0]] == 'njrat':
				row[-1] = '5'
			elif hashname[row[0]] == 'Trickbot':
				row[-1] = '6'

			
			i = i + 1				



			# feature_csv[i].extend(line.split(',')[0:-1])
			# if line.split(',')[-1] == '0':
			# 	feature_csv[i].append('0')
			# if line.split(',')[-1] == '1':
			# 	if hashname[line.split(',')[0]] == 'Gozi': 
			# 		feature_csv[i].append('1')
			# 	elif hashname[line.split(',')[0]] == 'GuLoader': 
			# 		feature_csv[i].append('2')
			# 	elif hashname[line.split(',')[0]] == 'Heodo': 
			# 		feature_csv[i].append('3')
			# 	elif hashname[line.split(',')[0]] == 'IcedID': 
			# 		feature_csv[i].append('4')
			# 	elif hashname[line.split(',')[0]] == 'njrat': 
			# 		feature_csv[i].append('5')
			# 	else:
			# 		hashname[line.split(',')[0]] == 'Trickbot'
			# 		feature_csv[i].append('6')
			


	csvfile = 'degree_multi_features.csv'
	with open(csvfile, 'w', newline='') as f:
		csvfile = csv.writer(f)
		csvfile.writerows(list_of_rows)


