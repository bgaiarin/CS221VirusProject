import csv
class mdp:

	def initState(self, responses_csv, initial_infections, initial_resources):
			state = [initial_resources]
			for country in self.countries:
				state = [0, 0] + state # 2 slots for each country, keeps resources at end

			with open(responses_csv) as responseFile:
				next(responseFile)
				csvReader = csv.reader(responseFile, delimiter=',')
				print self.countries
				for row in csvReader:
					country = row[0]
					print country
					if country in self.countries:
						print float(row[1])
						state[self.countries.index(country) + len(self.countries)] = float(row[1]) / self.RESPONSE_DENOMINATOR
						print "added country rank. new state ", state
					else:
						print "ERROR COUNTRY NOT FOUND IN DATABASE: ", country

			for country, infection in initial_infections.items():
				index = self.countries.index(country)
				if index >= self.NUM_COUNTRIES or index < 0:
					print "ERROR INITIALIZING STATE. COUNTRY NOT FOUND: ", country, index
				else:
					state[self.countries.index(country)] = infection
			print 'finished initializing state: ', state
			return state

	def __init__(self, transitions_csv, responses_csv, initial_infections = {}, initial_resources = 0):
		#self.countries, self.neighbors, self.totalSeats = loadFlights(transitions_csv)
		self.countries = ['France', 'South Africa', 'Mauritius', 'Nigeria']
		self.NUM_COUNTRIES = len(self.countries)
		self.INDEX_RESOURCE = self.NUM_COUNTRIES * 2
		self.RESPONSE_DENOMINATOR = 110.0 # amount response ranking is divided by during parsing; should be > 100
		self.INFECTION_COEFFICIENT = 3.0
		self.PREVENTION_COST = 0.8
		self.INFECTION_COST = 0.6 # 0 < x < 1, should be <= PREVENTION_COST
		self.MAX_RESPONSE_SCORE = 0.99
		self.NO_VIRUS_REWARD = 100.0 + self.NUM_COUNTRIES
		self.END_RESOURCES_WEIGHT = 10.0
		self.RESOURCES_DEPLETED_REWARD = -100.0 - (self.NUM_COUNTRIES * self.END_RESOURCES_WEIGHT)
		self.state = self.initState(responses_csv, initial_infections, initial_resources)

responses_csv = '../data/FR_MAUR_NIG_SA_responseIndicators.csv'
transitions_csv = '../data/FR_MAUR_NIG_SA_transitions.csv'
infections = {'France': 1, 'Mauritius': 0, 'Nigeria': 1}#, 'South Africa': 1}
mymdp = mdp(transitions_csv, responses_csv, infections, 13)



