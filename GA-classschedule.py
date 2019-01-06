from classDef import *
import random
import pandas
import numpy
import operator

class Chromosome(object):
	"""docstring for Chromosome"""
	def __init__(self, length):
		# print "sdsds"
		# print length
		self.genes=length
		self.geneArray = []

	def initChromosom(self):
		for i in range(0,self.genes):
			self.geneArray.append([])

	def __str__(self):
		return str(self.geneArray)

class fitness(object):
	"""docstring for fitness"""
	def __init__(self, individual):
		self.schedule = individual
		self.score=0

	def computeScore(self):
		score=0
		coursearray=self.schedule.getGeneArray()

		for i in coursearray:
			if len(i)==0:
				score+=5
				continue
			#check for multiple classes
			if(len(i)==1):
				score+=1
			# #check for equipment	
			# flag=0
			# for j in i:
			# 	if (j.assignedcourse.required != roomlist[coursearray.index(i)%len(roomlist)].equip):
			# 		flag=1
			# if (flag==0):
			# 	score+=1
			#check for seat number
			flag=0
			for k in i:
				if (k.group.size == roomlist[coursearray.index(i)%len(roomlist)].size):
					flag=1
			if (flag==0):
				score+=1
			#check for clashes
			temp=[]
			flag=0
			for m in i:
				if (m.group.id in temp):
					flag=1
				else:
					temp.append(m.group.id)
			if flag==0:
				score+=1

			#check for professor classes
			temp=[]
			flag=0
			for m in i:
				if (m.professor in temp):
					flag=1
				else:
					temp.append(m.professor)
			if flag==0:
				score+=1
		# print "score"
		# print score
		# print "opt"
		# print str(5*self.schedule.getlength())
		fitness=float(float(score)/(5.0*float(self.schedule.getlength())))
		#print "fit"
		#print fitness
		return fitness


class Schedule(object):
	"""docstring for Schedule"""
	def __init__(self, length):
		self.chromosome=Chromosome(length)
		self.chromosome.initChromosom()
		self.length=length
		# print "sdsdsdsdsd"
		# print str(self.chromosome.genes)

	def initialize(self):
		#print "sdsd"
		counter=0
		for i in cclist:
			for j in range(i.assignedcourse.slots):
				# print "fffffff"
				# print str(self.length)
				counter+=1
				#print "in"
				self.chromosome.geneArray[random.randrange(0, self.chromosome.genes)].append(i)
		# print "counter"
		# print counter

	def populateSchedule(self,hashtable):
		for i in hashtable.keys():
			# print "ga"
			# print self.chromosome.geneArray
			#print "length"
			#print self.chromosome.genes
			#print i
			self.chromosome.geneArray[(i/10)].append(hashtable[i])

	def getGeneArray(self):
		return self.chromosome.geneArray
	def getlength(self):
		return self.chromosome.genes

	def __repr__(self):
		#displaylist=[]
		return str(self.chromosome.geneArray)

def hashing(sch):
	hashtable={}
	genes=sch.getGeneArray()
	for i in genes:
		for j in range(len(i)):
			hashtable[(genes.index(i)+(10*j))]=i[j]
	return hashtable

def hashReverse(schHash,length):
	sch = Schedule(length)
	sch.populateSchedule(schHash)
	return sch

def createInitialPopulation(length, generationSize):
	generation=[]
	for i in range(0, generationSize):
		#print "ds"
		sch=Schedule(length)
		sch.initialize()
		generation.append(sch)
	return generation

def rank(population):
	# print "cp"
	# print population
	fitnessResults = {}
	for i in range(0,len(population)):
		fitnessResults[i] = fitness(population[i]).computeScore()
		print "pop score"
		print fitnessResults[i]
	return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def tournamentSelection(rankedPop, eliteSize):
	selectionPool = []
	dataframe = pandas.DataFrame(numpy.array(rankedPop), columns=["Index","Fitness"])
	dataframe['cumlative_sum'] = dataframe.Fitness.cumsum()
	dataframe['cumlative_percentage'] = 100*dataframe.cumlative_sum/dataframe.Fitness.sum()
	#print dataframe

	# print dataframe
	for i in range(0, eliteSize):
		selectionPool.append(rankedPop[i][0])
	for i in range(0, len(rankedPop) - eliteSize):
		threshold = 100*random.random()
		# print "thres"
		# print threshold
		for i in range(0, len(rankedPop)):
			if threshold <= dataframe.iat[i,3]:
				selectionPool.append(rankedPop[i][0])
				#print "a7aa"
				break
	#print "tour"
	#print len(rankedPop)
	return selectionPool

def matingPool(population, selectionPool):
    matingpool = []
    for i in range(0, len(selectionPool)):
        index = selectionPool[i]
        matingpool.append(population[index])
    return matingpool

def breed(schParentA, schParentB):
	print "bbbbbbbb"
	print schParentA
	print schParentB
	child = {}
	PAhash = hashing(schParentA)
	PBhash = hashing(schParentB)
	childP1={}
	childP2={}
	childP3={}

	geneA = int(random.random() * schParentA.getlength())
	geneB = int(random.random() * schParentA.getlength())

	startGene = min(geneA, geneB)
	endGene = max(geneA, geneB)

	# print startGene
	# print endGene
	childP1={k: PAhash[k] for k in PAhash.keys()[:startGene]}
	childP2={k: PBhash[k] for k in PBhash.keys()[startGene:endGene+1]}
	childP3={k: PAhash[k] for k in PAhash.keys()[endGene+1:]}
	# print "childs"
	# print type(childP1)
	# print ""
	# print type(childP2)
	# print ""
	# print type(childP3)
	# print ""
	# print type(child)

	child.update(childP1)
	child.update(childP2)
	child.update(childP3)
	childSch = hashReverse(child,schParentA.getlength())

	return childSch

def crossover(matingpool, eliteSize):
	print "cross"
	print matingpool
	nextGen = []
	randPool = random.sample(matingpool, len(matingpool))

	for i in range(0,eliteSize):
		nextGen.append(matingpool[i])

	for i in range(0, len(matingpool) - eliteSize):
		# print "iiiii"
		# print i
		child = breed(matingpool[i], matingpool[len(matingpool)-i-1])
		nextGen.append(child)
	# print "cross2"
	# print nextGen
	return nextGen

def mutate(schHash, mutationRate, mutationSize):
	for i in range(len(schHash)):
		if(random.random() < mutationRate):
			for j in range(0,mutationSize):
				swapWith = int(random.random() * len(schHash))

				course1 = schHash[schHash.keys()[i]]
				course2 = schHash[schHash.keys()[swapWith]]

				schHash[schHash.keys()[i]] = course2
				schHash[schHash.keys()[swapWith]] = course1
	return schHash

def mutatePop(population, mutationRate, mutationSize):
	# print "muin"
	# print population
	mutatedPop = []

	for i in range(0, len(population)):
		# print "poppppppp"
		# print population[i]
		mutatedHash=hashing(population[i])
		# print "mut"
		# print mutatedHash
		mutated = mutate(mutatedHash, mutationRate, mutationSize)
		mutatedPop.append(mutated)

	#print "mut"
	#print len(mutatedPop)
	return mutatedPop


def getNextGen(currentGen, eliteSize, mutationRate, mutationSize):
	popRanked = rank(currentGen)
	# print "step1"
	# print popRanked
	selectionResults = tournamentSelection(popRanked, eliteSize)
	# print "step2"
	# print selectionResults
	matingpool = matingPool(currentGen, selectionResults)
	print "sdsdsdsdsdsdsdsd"
	print matingpool
	children = crossover(matingpool, eliteSize)
	# print"cssss"
	# print children
	nextGenHashes = mutatePop(children, mutationRate, mutationSize)
	# print "hshshs"
	# print nextGenHashes
	nextGeneration=[]
	for i in nextGenHashes:
		# print "sdsdsdsdsd"
		# print len(i)
		individual=hashReverse(i,len(currentGen))
		# print "ind"
		# print individual
		nextGeneration.append(individual)
	# print "next"
	# print nextGeneration
	return nextGeneration

def displayPop(population):
	displaylist=[]
	for i in population:
		displaylist.append(i)
	print displaylist

def geneticAlgorithm(popSize, days, slots, rooms, eliteSize, mutationRate, mutationSize, generations):
	length=days*slots*rooms
	pop = createInitialPopulation(length, popSize)
	#print "init"
	#displayPop(pop)
	# print len(pop)
	# print "sds"
	# print rank(pop)
	print("Initial Score: " )+ str((rank(pop)[0][1]))

	for i in range(0, generations):
		pop = getNextGen(pop, eliteSize, mutationRate, mutationSize)
		#print rank(pop)
		#dataframe = pandas.DataFrame(numpy.array(pop))
		#print dataframe
		#print rank(pop)[0][1]
		#print "pop" + str(i)

	print("Final Score: " + str(rank(pop)[0][1]))
	bestScheduleIndex = rank(pop)[0][0]
	bestSchedule = pop[bestScheduleIndex]
	return bestSchedule

geneticAlgorithm(popSize=10,days=2,slots=2,rooms=2, eliteSize=0, mutationRate=0.01, mutationSize=1,generations=10)









