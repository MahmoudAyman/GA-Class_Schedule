from classDef import *
import random
import pandas
import numpy
import operator
import matplotlib.pyplot as plt

class Chromosome(object):
	"""docstring for Chromosome"""
	def __init__(self, length):
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
			flag=0
			for j in i:
				if (j.assignedcourse.required != roomlist[coursearray.index(i)%len(roomlist)].equip):
					flag=1
			if (flag==0):
				score+=1
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
		fitness=float(float(score)/(5.0*float(self.schedule.getlength())))
		return fitness


class Schedule(object):
	"""docstring for Schedule"""
	def __init__(self, length):
		self.chromosome=Chromosome(length)
		self.chromosome.initChromosom()
		self.length=length

	def initialize(self):
		# counter=0
		for i in cclist:
			for j in range(i.assignedcourse.slots):
				# counter+=1
				self.chromosome.geneArray[random.randrange(0, self.chromosome.genes)].append(i)

	def populateSchedule(self,hashtable):
		for i in hashtable.keys():
			temp=str(i)[-1]
			temp2=int(temp)
			self.chromosome.geneArray[(temp2)].append(hashtable[i])

	def getGeneArray(self):
		return self.chromosome.geneArray
	def getlength(self):
		return self.chromosome.genes

	def __repr__(self):
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
		sch=Schedule(length)
		sch.initialize()
		generation.append(sch)
	return generation

def rank(population):
	fitnessResults = {}
	for i in range(0,len(population)):
		fitnessResults[i] = fitness(population[i]).computeScore()
	return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def tournamentSelection(rankedPop, eliteSize):
	selectionPool = []
	dataframe = pandas.DataFrame(numpy.array(rankedPop), columns=["Index","Fitness"])
	dataframe['cumlative_sum'] = dataframe.Fitness.cumsum()
	dataframe['cumlative_percentage'] = 100*dataframe.cumlative_sum/dataframe.Fitness.sum()
	#print dataframe

	for i in range(0, eliteSize):
		selectionPool.append(rankedPop[i][0])
	for i in range(0, len(rankedPop) - eliteSize):
		threshold = 100*random.random()
		for i in range(0, len(rankedPop)):
			if threshold <= dataframe.iat[i,3]:
				selectionPool.append(rankedPop[i][0])
				break
	return selectionPool

def matingPool(population, selectionPool):
    matingpool = []
    for i in range(0, len(selectionPool)):
        index = selectionPool[i]
        matingpool.append(population[index])
    return matingpool

def breed(schParentA, schParentB):
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

	childP1={k: PAhash[k] for k in PAhash.keys()[:startGene]}
	childP2={k: PBhash[k] for k in PBhash.keys()[startGene:endGene+1]}
	childP3={k: PAhash[k] for k in PAhash.keys()[endGene+1:]}

	child.update(childP1)
	child.update(childP2)
	child.update(childP3)
	childSch = hashReverse(child,schParentA.getlength())

	return childSch

def crossover(matingpool, eliteSize):
	nextGen = []
	randPool = random.sample(matingpool, len(matingpool))

	for i in range(0,eliteSize):
		nextGen.append(matingpool[i])

	for i in range(0, len(matingpool) - eliteSize):
		child = breed(matingpool[i], matingpool[len(matingpool)-i-1])
		nextGen.append(child)
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
	mutatedPop = []

	for i in range(0, len(population)):
		mutatedHash=hashing(population[i])
		mutated = mutate(mutatedHash, mutationRate, mutationSize)
		mutatedPop.append(mutated)
	return mutatedPop


def getNextGen(currentGen, eliteSize, mutationRate, mutationSize):
	popRanked = rank(currentGen)
	selectionResults = tournamentSelection(popRanked, eliteSize)
	matingpool = matingPool(currentGen, selectionResults)
	children = crossover(matingpool, eliteSize)
	nextGenHashes = mutatePop(children, mutationRate, mutationSize)
	nextGeneration=[]
	for i in nextGenHashes:
		individual=hashReverse(i,len(currentGen))
		nextGeneration.append(individual)
	return nextGeneration

def displayPop(population):
	displaylist=[]
	for i in population:
		displaylist.append(i)
	print displaylist

def geneticAlgorithm(popSize, days, slots, rooms, eliteSize, mutationRate, mutationSize, generations):
	length=days*slots*rooms
	pop = createInitialPopulation(length, popSize)
	print("Initial Score: " )+ str((rank(pop)[0][1]))
	displayPop(pop)

	for i in range(0, generations):
		pop = getNextGen(pop, eliteSize, mutationRate, mutationSize)
		print pop

	print("Final Score: " + str(rank(pop)[0][1]))
	bestScheduleIndex = rank(pop)[0][0]
	bestSchedule = pop[bestScheduleIndex]
	return bestSchedule

def geneticAlgorithmPlot(popSize, days, slots, rooms, eliteSize, mutationRate, mutationSize, generations):
	length=days*slots*rooms
	pop = createInitialPopulation(length, popSize)
	progress = []
	popScore=(rank(pop)[0][1])
	print("Initial Score: " ) + str(popScore)
	progress.append(popScore)

	for i in range(0, generations):
	    pop = getNextGen(pop, eliteSize, mutationRate, mutationSize)
	    progress.append(rank(pop)[0][1])


	plt.plot(progress)
	plt.ylabel('fitness')
	plt.xlabel('Generation')
	plt.show()

geneticAlgorithm(popSize=10,days=2,slots=3,rooms=2, eliteSize=1, mutationRate=0.01, mutationSize=1,generations=10)









