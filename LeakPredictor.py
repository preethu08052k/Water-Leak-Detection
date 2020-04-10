from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import LeakDetection

class Calculator():

	def __init__(self):
		self.window=Tk()
		self.window.title("Leak Predictor")
		self.NF=Frame(self.window)
		self.NF.grid(row=0,column=0,columnspan=5,padx=3,pady=3)
		self.L1=Label(self.NF,text="Pulse Rate :")
		self.L1.grid(row=0,column=1,sticky=E)	
		self.L2=Label(self.NF,text="Quantity :")
		self.L2.grid(row=1,column=1,sticky=E)
		self.L3=Label(self.NF,text="Prediction :")
		self.L3.grid(row=2,column=1,sticky=E)
		self.L4=Label(self.NF,text="Accuracy :")
		self.L4.grid(row=3,column=1,sticky=E)
		self.E1=Entry(self.NF)
		self.E1.grid(row=0,column=2)	
		self.E2=Entry(self.NF)
		self.E2.grid(row=1,column=2)	
		self.E3=Entry(self.NF)
		self.E3.grid(row=2,column=2)
		self.E4=Entry(self.NF)
		self.E4.grid(row=3,column=2)
		self.BF=Frame(self.window)
		self.BF.grid(row=5,column=0,columnspan=6,padx=3,pady=3)
		self.B1=Button(self.BF,text="Logistic Regression",pady=2,padx=5,bg="lightblue",command=self.lr)
		self.B1.grid(row=0,column=0,pady=2,padx=5)
		self.B2=Button(self.BF,text="K Nearest Numbers",pady=2,padx=5,bg="lightblue",command=self.knn)
		self.B2.grid(row=0,column=1,pady=2,padx=5)
		self.B3=Button(self.BF,text="Support Vector Machines",pady=2,padx=5,bg="lightblue",command=self.svm)
		self.B3.grid(row=0,column=2,pady=2,padx=5)
		self.B4=Button(self.BF,text="Naive Bayes",pady=2,padx=5,bg="lightblue",command=self.nb)
		self.B4.grid(row=0,column=3,pady=2,padx=5)
		self.B5=Button(self.BF,text="Decision Tree",pady=2,padx=5,bg="lightblue",command=self.dt)
		self.B5.grid(row=0,column=4,pady=2,padx=5)
		self.B5=Button(self.BF,text="Clear",pady=2,padx=5,bg="lightblue",command=self.clear)
		self.B5.grid(row=0,column=5,pady=2,padx=5)
		self.window.mainloop()

	def lr(self):
		if(self.E1.get()=="" or self.E2.get()==""):
			messagebox.showinfo('Warning!','Please enter values')
			return
		pdf=DataFrame({'Pulse_Rate':[self.E1.get()],'Quantity':[self.E2.get()]})
		pred=LeakDetection.logregmodel.predict(pdf)[0]
		self.E3.delete(0,len(self.E3.get()))
		self.E4.delete(0,len(self.E4.get()))
		if pred==0:
			self.E3.insert(0,'NO LEAK')
		elif pred==1:
			self.E3.insert(0,'LEAK')
		self.E4.insert(0,str(round(LeakDetection.lrscore*100,2))+'%')

	def knn(self):
		if(self.E1.get()=="" or self.E2.get()==""):
			messagebox.showinfo('Warning!','Please enter values')
			return
		pdf=DataFrame({'Pulse_Rate':[self.E1.get()],'Quantity':[self.E2.get()]})
		pred=LeakDetection.knnmodel.predict(pdf)[0]
		self.E3.delete(0,len(self.E3.get()))
		self.E4.delete(0,len(self.E4.get()))
		if pred==0:
			self.E3.insert(0,'NO LEAK')
		elif pred==1:
			self.E3.insert(0,'LEAK')
		self.E4.insert(0,str(round(LeakDetection.knnscore*100,2))+'%')

	def svm(self):
		if(self.E1.get()=="" or self.E2.get()==""):
			messagebox.showinfo('Warning!','Please enter values')
			return
		pdf=DataFrame({'Pulse_Rate':[self.E1.get()],'Quantity':[self.E2.get()]})
		pred=LeakDetection.svmmodel.predict(pdf)[0]
		self.E3.delete(0,len(self.E3.get()))
		self.E4.delete(0,len(self.E4.get()))
		if pred==0:
			self.E3.insert(0,'NO LEAK')
		elif pred==1:
			self.E3.insert(0,'LEAK')
		self.E4.insert(0,str(round(LeakDetection.svmscore*100,2))+'%')

	def nb(self):
		if(self.E1.get()=="" or self.E2.get()==""):
			messagebox.showinfo('Warning!','Please enter values')
			return
		pdf=DataFrame({'Pulse_Rate':[self.E1.get()],'Quantity':[self.E2.get()]})
		pred=LeakDetection.nbmodel.predict(pdf)[0]
		self.E3.delete(0,len(self.E3.get()))
		self.E4.delete(0,len(self.E4.get()))
		if pred==0:
			self.E3.insert(0,'NO LEAK')
		elif pred==1:
			self.E3.insert(0,'LEAK')
		self.E4.insert(0,str(round(LeakDetection.nbscore*100,2))+'%')

	def dt(self):
		if(self.E1.get()=="" or self.E2.get()==""):
			messagebox.showinfo('Warning!','Please enter values')
			return
		pdf=DataFrame({'Pulse_Rate':[self.E1.get()],'Quantity':[self.E2.get()]})
		pred=LeakDetection.dtmodel.predict(pdf)[0]
		self.E3.delete(0,len(self.E3.get()))
		self.E4.delete(0,len(self.E4.get()))
		if pred==0:
			self.E3.insert(0,'NO LEAK')
		elif pred==1:
			self.E3.insert(0,'LEAK')
		self.E4.insert(0,str(round(LeakDetection.dtscore*100,2))+'%')

	def clear(self):
		if messagebox.askquestion('Clear','Are you sure you want to clear?'):
			self.E1.delete(0,len(self.E1.get()))
			self.E2.delete(0,len(self.E2.get()))
			self.E3.delete(0,len(self.E3.get()))
			self.E4.delete(0,len(self.E4.get()))

if __name__=="__main__":
	Calculator()