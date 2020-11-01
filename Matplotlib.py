import matplotlib.pyplot as plt
import numpy as np

#1 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(1, 49, 10)
ax.plot(x, 3 * x, 'b')
ax.axis([0, 50, 0, 160])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Draw a line.")
plt.show()

#2
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(1, 3, 4)
ax.plot(x, 2 * x, 'b', x, -3 * x + 10, 'b')
ax.axis([1, 3, 1, 4])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Sample graph!")
plt.show()

# 3
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(10, 30, 4)
ax.plot(x, -3 * x + 100, 'b', label='line 1')
ax.plot(x, 2 * x - 30, 'g', label='line 2')
ax.plot(x, 2 * x, 'b')
ax.plot(x, -3 * x + 70, 'g')
plt.legend()
ax.axis([10, 30, 10, 40])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Two or more lines on same plot with suitable legend!")
plt.show()

# 4
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(10, 30, 4)
ax.plot(x, -3 * x + 100, 'b', linewidth=3, label='line1-width-3')
ax.plot(x, 2 * x - 30, 'r', linewidth=5, label='line2-width-5')
ax.plot(x, 2 * x, 'b', linewidth=3)
ax.plot(x, -3 * x + 70, 'r', linewidth=5)
plt.legend()
ax.axis([10, 30, 10, 40])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Two or more lines with different width and colors with suitable legend")
plt.show()

#5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(10, 30, 4)
ax.plot(x, -3 * x + 100, 'b',ls='dotted', linewidth=3, label='line1-dotted')
ax.plot(x, 2 * x - 30, 'r',ls='dashed', linewidth=5, label='line2-dashed')
ax.plot(x, 2 * x, 'b', ls='dotted',linewidth=3)
ax.plot(x, -3 * x + 70, 'r',ls='dashed', linewidth=5)
plt.legend()
ax.axis([10, 30, 10, 40])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Plot with two or more lines with different styles")
plt.show()

#6
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
ax.plot([1,4,5,6,7],[2,6,3,6,3], 'r',ls='-.', linewidth=1.5) 
ax.plot([1,4,5,6,7],[2,6,3,6,3],'ob')
ax.axis([1, 8, 1, 8])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Display marker")
plt.show()

#7
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(1, 49, 10)
ax.axis([0, 50, 0, 150])
ax.plot(x, 3 * x, 'b')
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Draw a line.")
ax.axis([0,100,0,200])
plt.show()

# 8
x1 = [2, 3, 5, 6, 8]
y1 = [1, 5, 10, 18, 20]
x2 = [3, 4, 6, 7, 9]
y2 = [2, 6, 11, 20, 22]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
ax.plot(x1, y1, '*b', linewidth=1.5)
ax.plot(x2, y2, 'or')
ax.axis([0, 10, 0, 30])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.title("Display marker")
plt.show()

# 9
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
x = np.linspace(0, 5, 20);
ax.plot(x, 5 * x ** 2, 'r^', x, x ** 2, 'bs', x, 3 * x / 2, 'g--')
ax.axis([0, 5, 0, 120])
plt.ylabel("y - axis")
plt.xlabel("x - axis")
plt.show()

# 10
import datetime as DT
from matplotlib.dates import date2num

data = [(DT.datetime.strptime('2016-10-03', "%Y-%m-%d"), 772.559998),
        (DT.datetime.strptime('2016-10-04', "%Y-%m-%d"), 776.429993),
        (DT.datetime.strptime('2016-10-05', "%Y-%m-%d"), 776.469971),
        (DT.datetime.strptime('2016-10-06', "%Y-%m-%d"), 776.859985),
        (DT.datetime.strptime('2016-10-07', "%Y-%m-%d"), 775.080017)]

x = [date2num(date) for (date, value) in data]
y = [value for (date, value) in data]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)
ax.plot(x, y, 'r-o')
ax.set_xticks(x)
ax.set_xticklabels([date.strftime("%Y-%m-%d") for (date, value) in data])
plt.xlabel('Date')
plt.ylabel('Closing Value')
plt.title('Closing stock value of Alphabet Inc.')
plt.grid(linestyle='-', linewidth='0.5', color='blue')
plt.show()

#11
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(axis = 'both',which = 'major',
               direction = 'in',
               bottom = True,top = True, left = True,right = True)

ax.plot(x,y,'r-o')
ax.set_xticks(x)
ax.set_xticklabels([date.strftime("%Y-%m-%d") for (date, value) in data])
plt.xlabel('Date')
plt.ylabel('Closing Value')
plt.title('Closing stock value of Alphabet Inc.')
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.tick_params(which='both',top='off',left='off', right='off',bottom='off')
plt.show()
