import os, csv

# dict of origins, to dicts of destination : total seat count
flights = {}

for filename in os.listdir('./'):
	if not filename.endswith('.py'):
		with open(filename) as dataFile:
			next(dataFile)
			csvReader = csv.reader(dataFile, delimiter=',')
			for row in csvReader:
				origin = 'United States'#row[0].split(':')[1]
				dest = row[9]
				size = int(row[11])

				if dest == origin:
					continue

				if origin not in flights.keys():
					flights[origin] = {}

				if dest not in flights[origin]:
					flights[origin][dest] = 0

				flights[origin][dest] += size

with open('domestic-transitions.csv', 'w+') as outFile:
	for origin, dests in flights.items():
		for dest, size in dests.items():
			outFile.write('' + origin + ',' + dest + ',' + str(size) + '\n')