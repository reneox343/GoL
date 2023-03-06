import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from numba import njit
import cv2 
import time
from lifeForms import *
from prettytable import PrettyTable,ALL
from datetime import date
ON = 255
OFF = 0    
vals = [ON, OFF]
def randomGrid(height,width):
    """returns a grid of NxN random values"""
    
    return np.random.choice(vals, width*height, p=[0.2, 0.8]).reshape(height, width)

def deadGrid(N,M):
    return np.zeros((N,M))


def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0,    0, ON], 
                       [ON,  0, ON], 
                       [0,  ON, ON]])
    grid[i:i+3, j:j+3] = glider
def addBlock(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    block = np.array([[ ON,  ON, 0], 
                       [ON,  ON, 0], 
                       [0,  0, 0]])
    grid[i:i+3, j:j+3] = block
def addTub(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    block = np.array([[ 0,  ON, 0], 
                       [ON,  0, ON], 
                       [0,  ON, 0]])
    grid[i:i+3, j:j+3] = block
def addBlinker(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    blinker = np.array([[0,  ON, 0], 
                       [0,  ON, 0], 
                       [0,  ON, 0]])
    grid[i:i+3, j:j+3] = blinker

@njit()
def checkNeighbors(grid,newGrid,heigth,width):
    #basic game of life rules
    for i in range(0, heigth):
        for j in range(0, width):
            cont = 0
            cell = grid[i, j] 
            for x in range(-1,2,1):
                for y in range(-1,2,1):
                    # print(i+x,j+y)
                    if(x == 0 and y == 0):
                        continue
                    if(i+x<0 or i+x>=heigth):
                        continue
                    if(j+y<0 or j+y>=width):
                        continue
                    if(grid[i+x, j+y] == ON):
                        cont+=1
            if(cont < 2 and cell==ON):
                newGrid[i, j] = OFF
            if(cont == 3 and cont == 4 and cell ==ON):
                newGrid[i,j] = ON
            if(cont > 3 and cell== ON):
                newGrid[i, j] = OFF
            if(cont == 3 and cell == OFF):
                newGrid[i, j] = ON

def findLife(grid,i):
    
    aux = np.pad(grid,2)
    #start counting time
    # start = time.time()
    #create table
    table = PrettyTable(["life Form", "Count","Percent"])
    #find still life forms
    blockCont = explore(aux,block)
    beehiveCont = explore(aux,beehive,2)
    loafCont = explore(aux,loaf,4)
    boatCont = explore(aux,boat,4)
    tubCont = explore(aux,tub)
    #find oscilators life forms
    blinkerCont = explore(aux,blinker,2)
    toadCont = explore(aux,toad01,2) + explore(aux,toad02,2)
    beaconCont1 = explore(aux,beacon01,2)
    beaconCont2 = explore(aux,beacon02,2)
    beaconCont = beaconCont1+beaconCont2
    #find gliders life forms
    gliderCont = explore(aux,glider01,4)+explore(aux,glider02,4)+explore(aux,glider03,4)+explore(aux,glider04,4)
    spaceCont = explore(aux,space01,4)+explore(aux,space02,4)+explore(aux,space03,4)+explore(aux,space04,4)
    #fix block counter if beacon exist
    blockCont = blockCont - (beaconCont1*2)
    total = blockCont+beehiveCont+loafCont+boatCont+tubCont+blinkerCont+toadCont+gliderCont+spaceCont
    fakeTotal = max(total,1)
    #format information
    table.add_row(["block",blockCont,int(blockCont/fakeTotal*100)])
    table.add_row(["beehive",beehiveCont,int(beehiveCont/fakeTotal*100)])
    table.add_row(["loaf",loafCont,int(loafCont/fakeTotal*100)])
    table.add_row(["boat",boatCont,int(boatCont/fakeTotal*100)])
    table.add_row(["tub",tubCont,int(tubCont/fakeTotal*100)])
    table.add_row(["blinker",blinkerCont,int(blinkerCont/fakeTotal*100)])
    table.add_row(["toad",toadCont,int(toadCont/fakeTotal*100)])
    table.add_row(["beacon",beaconCont,int(beaconCont/fakeTotal*100)])
    table.add_row(["glider",gliderCont,int(gliderCont/fakeTotal*100)])
    table.add_row(["space",spaceCont,int(spaceCont/fakeTotal*100)])
    table.add_row(["total",total,"-"])
    #print report in file
    output = open(r"output.out", "a")
    output.write(f"iteration: {i}\n")
    output.write(f"{table}\n\n")
    #print repor on console
    print(f"iteration: {i}")
    print(table)
    #mesure time
    # end = time.time()
    # print(end - start)

#searches for the live forms
def explore(grid,form,rotations =0):
    #this affects how well the algoritm can see figures
    threshold = 0.5
    if(rotations == 0):
        res = cv2.matchTemplate(grid,form,cv2.TM_SQDIFF)
        loc = np.where(res <= threshold)
        return len(loc[0])
    else:
        #this rotates the life forms if needed
        newform = form.copy()
        cont = 0
        for i in range(rotations):
            res = cv2.matchTemplate(grid,newform,cv2.TM_SQDIFF)
            loc = np.where(res <= threshold)
            cont += len(loc[0])
            newform = np.rot90(newform)
        return cont

def update(i, img, grid, height,width):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line 
    i+=1
    newGrid = grid.copy()
    # TODO: Implement the rules of Conway's Game of Life
    checkNeighbors(grid,newGrid,height,width)
    #finds the life forms in the grid
    findLife(grid,i)
    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,

# main() function
def main():
    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life system.py")
    # TODO: add arguments
    
    # set grid size
    print("select 1 for random matrix")
    print("select 2 for reading an input file")
    option = int(input())
    # declare grid
    grid = np.array([])
    # set animation update interval
    updateInterval = 100
    iterations = 200
    if(option == 1):
        print("input height: ")
        height = int(input())
        print("input width: ")
        width = int(input())
        print("input iterations:")
        iterations = int(input())
        # populate grid with random on/off - more off than on
        grid  = randomGrid(height,width)
        # grid  = deadGrid(N,N)
    if(option == 2):
        width,height,iterations,grid = inputGrid()

    #changes the type of data of the grid so i can work with it
    grid = grid.astype(np.uint8)
    #prepare output file
    output = open(r"output.out", "w+")
    output.close()
    output = open(r"output.out", "a")
    output.write(f"Simulation at {date.today()} \nUniverse size {width}x{height}\n\n")
    output.close()

    # set up animation
    fig, ax = plt.subplots()
    # img = ax.imshow(grid, interpolation='nearest')
    img = ax.imshow(grid)
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, height,width),repeat =False,frames=iterations,interval=updateInterval)
    plt.show()

def inputGrid():
    file = open("input.in","r").readlines()
    #reads width and height
    width,height = map(int,str(file[0]).replace("\n","").split(" "))
    #read the number of iterations
    iterations = int(str(file[1]).replace("\n",""))
    #creates a grid
    grid = np.zeros((height,width))
    for x in range(2,len(file)):
        #reads rows and then cols i did it this way becuase of we read first width and then height
        j,i = map(int,str(file[x]).replace("\n","").split(" "))
        #fixex if needed the inputs
        if(j>=width):
            j = width-1
        if(i>=height):
            i = height-1
        grid[i][j] = ON
    return width,height,iterations,grid
    

# call main
if __name__ == '__main__':
    main()