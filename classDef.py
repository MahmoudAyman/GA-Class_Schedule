import numpy
import itertools

class Professor(object):
	"""docstring for Professor"""
	name=""
	courses=[]
	def __init__(self, profName, courseList):
		self.name=profName
		self.courses=courseList

	def __str__(self):
		return name

class Classroom(object):
	"""docstring for Classroom"""
	def __init__(self, classNumber, chairs, isEquip):
		self.id=classNumber
		self.size=chairs
		self.equip=isEquip

	def __str__(self):
		return str(self.id)

class Course(object):
	"""docstring for Course"""
	def __init__(self, courseTitle, hours,needEquip):
		self.name=courseTitle
		self.slots=hours
		self.required=needEquip

	def __repr__(self):
		return str(self.name)

class StudentGroup(object):
	"""docstring for StudentGroup"""
	courses=[]
	def __init__(self, number, students, courselist):
		self.id=number
		self.size=students
		self.courses=courselist
		

class CourseClass(object):
	"""docstring for ClassName"""
	objectId = itertools.count().next
	def __init__(self, groupName,course, professorID):
		self.group = groupName
		self.assignedcourse=course
		self.professor=professorID

	def __repr__(self):
		# print "st"
		return str(self.assignedcourse)[0:3]

	def __getitem__(self):
		print "ge"
		return self.group


#creating test data
C1=Course("Intro to algorithms", 3, True)
C2=Course("English", 1,False)
C3=Course("Chemistry", 3,True)
C4=Course("Calculs III", 4,False)
C5=Course("Humanities 101", 1,False)
C6=Course("Physics", 2,True)
C7=Course("Software Engineering", 2,False)
C8=Course("Circuits 1", 3,True)
C9=Course("Differntial Equations", 4,False)
C10=Course("Data Analysis", 2, True)
C11=Course("MATH 1", 3, True)
C12=Course("MATH 2", 1,False)
C13=Course("MATH 3", 3,True)
C14=Course("Physics 2", 4,False)
C15=Course("photography 1", 1,False)
C16=Course("photography 2", 2,True)
C17=Course("Software Engineering", 2,False)
C18=Course("Data Structure", 3,True)
C19=Course("Computer Architecture", 4,False)
C20=Course("Philosophy", 2, True)

P1=Professor("Ahmed",[C1,C2])
P2=Professor("Mohamed",[C3,C4])
P3=Professor("Mahmoud",[C5])
P4=Professor("Hussien",[C6,C9,C7])
P5=Professor("Ibrahim",[C10,C8])
P6=Professor("Ali",[C1,C2])
P7=Professor("Foaad",[C3,C4])
P8=Professor("Abduall",[C5])
P9=Professor("Abdelrahman",[C6,C9,C7])
P10=Professor("Shawky",[C10,C8])

CR1=Classroom(1,30,True)
CR2=Classroom(2,10,False)
CR3=Classroom(3,20,True)
CR4=Classroom(4,50,False)
CR5=Classroom(5,10,True)
# CR6=Classroom(6,20,False)

SG1=StudentGroup(1,25,[C1,C3,C5])
SG2=StudentGroup(2,40,[C4,C1])
SG3=StudentGroup(3,20,[C9,C10,C8])
SG4=StudentGroup(4,9,[C2,C3,C6])
SG5=StudentGroup(5,25,[C1,C3,C5])
SG6=StudentGroup(6,5,[C4,C1])
SG7=StudentGroup(7,22,[C9,C10,C8])
SG8=StudentGroup(8,9,[C2,C3,C6])

CC1=CourseClass(SG1,C1,P1)
CC2=CourseClass(SG1,C2,P2)
CC3=CourseClass(SG2,C3,P2)
CC4=CourseClass(SG2,C4,P3)
CC5=CourseClass(SG3,C5,P4)
CC6=CourseClass(SG3,C6,P5)
CC7=CourseClass(SG4,C7,P3)
CC8=CourseClass(SG4,C8,P6)
CC9=CourseClass(SG5,C9,P10)
CC10=CourseClass(SG5,C10,P3)
CC11=CourseClass(SG6,C1,P1)
CC12=CourseClass(SG6,C2,P6)
CC13=CourseClass(SG7,C3,P2)
CC14=CourseClass(SG7,C4,P7)
CC15=CourseClass(SG8,C5,P4)
CC16=CourseClass(SG8,C6,P5)
CC17=CourseClass(SG3,C7,P9)
CC18=CourseClass(SG2,C8,P9)
CC19=CourseClass(SG6,C9,P1)
CC20=CourseClass(SG7,C10,P3)

roomlist=[CR1,CR2,CR3,CR4,CR5]
#cclist=[CC1,CC2,CC3,CC4,CC5,CC6,CC7,CC8,CC9,CC10,CC11,CC12,CC13,CC14,CC15,CC16,CC17,CC18,CC19,CC20]
cclist=[CC1,CC2,CC3,CC4,CC5,CC6,CC7,CC8,CC9,CC10]
		
		


		
		