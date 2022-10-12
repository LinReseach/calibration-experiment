#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:41:12 2022

@author: chenglinlin
"""
"""
We want to have a continuous pictures so we make picture taking and present dots separate.
In consequence. In this calibration.py, we only present dots, don’t to take pictures.
Present dots firstly,then followed with arrows -2022/10/04
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 13:04:52 2019

@author: User1
"""
import numpy as np
import pandas as pd
import time
import datetime
import cv2
import os
from psychopy import visual, core, event, monitors

import socket
import cv2
import numpy as np
from PIL import Image
import glob


cols = ['frameNr','x','y','dotNr','arrowOri','Resp', 'corrResp','fName', 'sampTime']# for pandas dataframe
headerInfo = pd.DataFrame([], columns=cols)

def getMac():
    from uuid import getnode as get_mac
    mac = ':'.join(['{:02x}'.format((get_mac() >> ele) & 0xff) for ele in range(0,8*6,8)][::-1])
    return mac

def isNumber(s):
    """
    Tests whether an input is a number

    Parameters
    ----------
    s : string or a number
        Any string or number which needs to be type-checked as a number

    Returns
    -------
    isNum : Bool
        Returns True if 's' can be converted to a float
        Returns False if converting 's' results in an error

    Examples
    --------
    >>> isNum = isNumber('5')
    >>> isNum
    True

    >>> isNum = isNumber('s')
    >>> isNum
    False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def drawText(win,
             text='No text specified!',
             textKey=['space'],
             wrapWidth=1800,#900rrrrrrrrrrrrrrrrr
             textSize=100,#100rr25rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
             textColor=[255, 255, 255]):#[0 0 0]rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    """
    Draw a string on a psychopy window and waits for a keypress, always tries
    to draw the text in the center of the screen.

    Parameters
    ----------
    win : psychopy window
        An instance of an active psychopy window on which to draw the text
    text : string
        The text to draw
    textKey : list
        A list of the allowed keys to press to exit the function. The function
        will block code execution until the specified key or escape is pressed
    wrapWidth : int
        The number of characters to display per line. If there are more
        characters on one line than specified in wrapWith the text will
        continue on the next line
    textSize : int
        The height of the text in pixels
    textColor : list of [R,G,B] values
        The color in which to draw the text, [R,G,B]

    Returns
    -------
    key : string
        The key pressed
    rt : float
        The time from text display onset until keypress in seconds

    Examples
    --------
    >>> key, rt = pl.drawText(win, 'Press "Space" to continue!')
    >>> key
    'space'
    >>> rt
    1.2606524216243997
    """

    if np.sum(np.array(textColor) == 0) == 3 and np.sum(win.color < 100) == 3:
        textColor = [255, 255, 255]

    # textDisp = visual.TextStim(win, text=text, wrapWidth=wrapWidth,rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    #                            height=textSize, colorSpace='rgb255',
    #                            color=textColor, pos = (650,0))
    
    textDisp = visual.TextStim(win, text=text, wrapWidth=wrapWidth,
                               height=textSize, colorSpace='rgb255',
                               color=textColor, pos = (0,0))#pos=(100,0)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    textDisp.draw()
    time = core.Clock()
    win.flip()
    if isNumber(textKey[0]):
        core.wait(textKey[0])
        key = ['NoKey']
    else:
        key = getKey(textKey)
    rt = time.getTime()
    #print(textColor)
    return key[0], rt

def getKey(allowedKeys=['left', 'right'], waitForKey=True, timeOut=0):
    """
    Gets a keypress by using the event.waitKeys or event.getKeys from
    the psychopy module

    The escape key is always allowed.

    Parameters
    ----------
    allowedKeys : list, list fo strings
        The list should contain all allowed keys
    waitForKey : Bool
        If True, the code waits until one of the keys defined in allowedkeys
        or escape has been pressed
    timeOut : int or float, positive value
        Only has effect if waitForKey == True\n
        If set to 0, the function waits until an allowed key is pressed\n
        If set to any other positive value, breaks after timeOut seconds

    Returns
    -------
    key_pressed : tuple with two items
        The first index returns the Key\n
        The second index returns the timestamp\n
        The timestamp is in seconds after psychopy initialization and does not
        reflect the duration waited for the key press\n
        If timeOut or no key is pressed, returns ['NoKey', 9999]

    Note
    --------
    The function requires an active psychopy window
R
    Examples
    --------
    >>> key = getKey(allowedKeys = ['left', 'right'], waitForKey = True, timeOut = 0)
    >>> key # the 'left' key is pressed after 156 seconds'
    ('left', 156.5626505338878)
    """
    if waitForKey:
        while True:
            # Get key
            if timeOut > 0:
                key_pressed = event.waitKeys(maxWait=timeOut, timeStamped=True)
                if key_pressed is None:
                    key_pressed = [['NoKey', 9999]]
                    break
            else:
                key_pressed = event.waitKeys(maxWait=float('inf'), timeStamped=True)
            # Check last key
            if key_pressed[-1][0] == 'escape':
                break
            if key_pressed[-1][0] in allowedKeys:
                break

    else:
        # Get key
        key_pressed = event.getKeys(allowedKeys, timeStamped=True)
        if not key_pressed:
            key_pressed = [['NoKey', 9999]]

    return key_pressed[-1]

#def makeSquareGrid(x=0, y=0, grid_dimXY=[10, 10], line_lengthXY=[10, 10]):rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
def makeSquareGrid(x=0, y=0, grid_dimXY=[20, 20], line_lengthXY=[20, 20]):

    """
    Creates the coordinates for a square grid.

    Parameters
    ----------
    x : float or int
        The center x position of the grid
    y : float or int
        The center y position of the grid
    grid_dimXY : list, positive integers
        The size of the grid, e.g. the number of points in each direction
    line_lengthXY : list, positive floats or ints
        The length between each grid intersection, [width, height]

    Returns
    -------
    gridpositions : list of tuples
        Each tuple contains the (x,y) position of one of the grid intersections

    Examples
    --------
    >>> gridpositions = makeSquareGrid(0,0,[4,4],[10,10])
    >>> gridpositions
    [(-15.0, -15.0),
     (-15.0, -5.0),
     (-15.0, 5.0),
     (-15.0, 15.0),
     (-5.0, -15.0),
     (-5.0, -5.0),
     (-5.0, 5.0),
     (-5.0, 15.0),
     (5.0, -15.0),
     (5.0, -5.0),
     (5.0, 5.0),
     (5.0, 15.0),
     (15.0, -15.0),
     (15.0, -5.0),
     (15.0, 5.0),
     (15.0, 15.0)]

    """
    # Left starting position
    start_x = x - 0.5 * grid_dimXY[0] * line_lengthXY[0] + 0.5 * line_lengthXY[0]
    # Top starting position
    start_y = y - 0.5 * grid_dimXY[1] * line_lengthXY[1] + 0.5 * line_lengthXY[1]
    # For loops for making grid
    gridpositions = []
    for x_count in range(0, grid_dimXY[0]):
        current_x = start_x + x_count * line_lengthXY[0]
        for y_count in range(0, grid_dimXY[1]):
            current_y = start_y + y_count * line_lengthXY[1]
            gridpositions.append((current_x, current_y))
    return gridpositions


def calibration(win, fileName, calibration, pc='default', nrPoints=9, dotColor=[255, 255, 255],flagwrite=1):# dotColor=[0 0 0];add flagwriterrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    """
    Custom calibration using psychoLink. It uses the background
    color which is set in win. Flips the screen empty before returning.
 
    Parameters
    ----------
    win : psychopy window
        An instance of an active psychopy window on which to draw the text
    nrPoints : int
        The number of calibration points to use, allowed input:\n
        9,13,15 or 25
    dotColor : list, [R,G,B]
        The RGB color of the validation dot

    Returns
    -------
    gridPoints : 2d np.array (col 1 x positions, col2 y positions)
 

    Examples
    --------

    """
    # Get required information from the supplied window
    xSize, ySize = win.size#win.size rrrrrrrrrrrrrrr
    bgColor = list(win.color)
    arrowColor = [255,255,255]#[255,0,0]rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    escapeKey = ['None']
    sampDur = 2000#2000rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    textColor = [255, 0, 0]#[0 0 0 ]rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    waitForSacc = 1.00
    radStep = 5#0.75rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    stepDir = -radStep
    dotRad = 60#10rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    maxDotRad = 15
    minDotRad = 15#3rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    nFramesPerDot = 1#30 rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    nRandDots = 81#60
    arrowLineW = 60 #revise1=30;arrowLineW=15 rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    leftRight = ['left', 'right']
    totalDots = nrPoints + nRandDots
    nFrames = totalDots*nFramesPerDot
    #cols = ['frameNr','x','y','dotNr','arrowOri','Resp', 'corrResp','fName', 'sampTime']# for pandas dataframe
    #headerInfo = pd.DataFrame([], columns=cols)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    headerInfo['frameNr'] = np.arange(1,nFrames+1)
    headerInfo['pc'] = pc
    headerInfo['resX'] = xSize
    headerInfo['resY'] = ySize
    #calDotNr = np.zeros(nFrames)
    
    if np.sum(np.array(textColor) == 0) == 3 and np.sum(win.color < 100) == 3:
        textColor = [255, 255, 255]
    if np.sum(np.array(dotColor) == 0) == 3 and np.sum(win.color < 100) == 3:
        dotColor = [255, 255, 255]
    print(dotColor)#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    # Initiate Dots (inner and outer dot for better fixation)
    OuterDot = visual.Circle(win,
                             radius=60,#revise1=20;radius=10  rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                             lineWidth=1,
                             fillColorSpace='rgb255',
                             lineColorSpace='rgb255',
                             lineColor=bgColor,
                             fillColor=dotColor,
                             edges=40,
                             pos=[0, 0])

    InnerDot = visual.Circle(win,
                             radius=1,#revised1=2;radius=1 rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                             lineWidth=1,
                             fillColorSpace='rgb255',
                             lineColorSpace='rgb255',
                             lineColor=bgColor,
                             fillColor=bgColor,
                             edges=40,
                             pos=[0, 0])
    
    arrowLine = visual.Rect(win,
                            width=arrowLineW,
                            height=arrowLineW/5, 
                            fillColorSpace='rgb255',
                            lineColorSpace='rgb255',
                            lineColor=arrowColor,
                            fillColor=arrowColor,
                            lineWidth=0,
                            pos=[0,0])
    
    arrowHead = visual.Polygon(win,
                               radius=arrowLineW/2,
                               fillColorSpace='rgb255',
                               lineColorSpace='rgb255',
                               lineColor=arrowColor,
                               fillColor=arrowColor,
                               lineWidth = 0,
                               pos=[0,0])
            
    def drawDots(point, rad, col         ):
        OuterDot.fillColor = col
        OuterDot.pos = point
        OuterDot.radius = rad
        OuterDot.draw()
        InnerDot.pos = point
        InnerDot.draw()

    def drawArrow(point, lr='left', col=[0,0,0]):
        if lr == "left": 
            arrowPoint = [point[0]-(arrowLineW/2), point[1]]
            ori = 270
        elif lr == 'right':
            arrowPoint =  [point[0]+(arrowLineW/2), point[1]]
            ori = 90
        arrowLine.fillColor = col
        arrowLine.pos = point
        arrowLine.draw()
        arrowHead.fillColor = col
        arrowHead.pos = arrowPoint
        arrowHead.ori = ori
        arrowHead.draw()

    # Make the grid depending on the number of points for calibration
    if nrPoints == 9:#it is critical to the whole code and need change;rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        # xlineLength = (xSize - 150) / 2
        # yLineLength = (ySize - 150) / 2
        xlineLength = (xSize - 150) / 2
        yLineLength = (ySize - 150) / 2
        gridPoints = makeSquareGrid(0, 0, [3, 3], [xlineLength, yLineLength])
        print(nrPoints,xSize)

    elif nrPoints == 13:
        xlineLength = (xSize - 150) / 2
        yLineLength = (ySize - 150) / 2
        gridPoints = makeSquareGrid(0, 0, [3, 3], [xlineLength, yLineLength])
        gridPoints += makeSquareGrid(0, 0, [2, 2], [xSize / 2, ySize / 2])

    elif nrPoints == 15:
        xlineLength = (xSize - 150) / 4
        yLineLength = (ySize - 150) / 2
        gridPoints = makeSquareGrid(0, 0, [5, 3], [xlineLength, yLineLength])

    elif nrPoints == 25:
        xlineLength = (xSize - 150) / 4
        yLineLength = (ySize - 150) / 4
        gridPoints = makeSquareGrid(0, 0, [5, 5], [xlineLength, yLineLength])

    else:
        drawText(win, 'Incorrect number of validation points,\n please try again with a different number', 3)
        win.flip()
        return headerInfo

    # shuffle points
    #np.random.shuffle(gridPoints)

    # remove the fixation posiiton from the gridpoints and add it as the last point
    # gridPoints = [i for i in gridPoints if i != (0.0, 0.0)]#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    # gridPoints.append((0, 0))
   
    gridPoints = [i for i in gridPoints*9]# gridPoints  rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    
    # Add the random dot locations rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    # randX = np.random.randint((-(xSize)/2) + 10,(xSize/2)-9, nRandDots)
    # randY = np.random.randint((-(ySize)/2) + 10,(ySize/2)-9, nRandDots)
    # gridPoints=[(0,0)]
    # for (x, y) in zip(randX,randY):
    #     gridPoints.append((x, y))
        
    # for (x, y) in zip(randX, randY):
    #     gridPoints.append((x, y))

    # Draw the first fixation dot and wait for spacepress to start validation
    startKey = drawText(win, 'Position1. Press space to start calibration!',['space', 'escape'])[0]
    #startKey = drawText(win, 'S',['space', 'escape'])[0]#rrrrrrrrrrrrrrr
    drawDots((0, 0), dotRad, dotColor)
    win.flip()
    time.sleep(sampDur/1000.)
    if startKey[0] == 'escape':
        escapeKey[0] = 'escape'
        return headerInfo
    
    print("I AM HERE2") #rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    
    # Determine if there is a face with 2 eyes in the video before starting
    # eyeFound = determineIfEye()#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    # if not eyeFound:
    #     return headerInfo
    
    # Draw the Dots dot and wait for 1 second between each dot
    fCount = 0
    ffCount=0
    aa=2
    ff=fCount#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    frame=0
    framegaze=0
    for i in range(0, len(gridPoints)): 
        
        s  = time.time()
        curFCount = 0
        dotRadius = dotRad
        
        # Draw arrow and wait for responsse
        lr = np.random.choice(2)
        respIdxStart = fCount # fCount rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        
        while True:
            frame=frame+1
            framegaze=framegaze+1
            fName = fileName+'/'+'%05d.jpg' % (ffCount+1)
            #cv2.imwrite(fName, getFrame())
            ffCount+=1
            print('frame',frame)
            
            if dotRadius > maxDotRad:
                stepDir = -radStep
            elif dotRadius < minDotRad:
                stepDir = radStep
            dotRadius += stepDir            
            drawDots(gridPoints[i], dotRadius, dotColor)
            sampTime = win.flip()
            if (time.time() - s) > waitForSacc:
           # if (time.time() - s) > waitForSacc:
                # Get video image
                #if flagwrite==1:
                    #print(nrPoints,xSize,'win.size=',win.size)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                    #fileName=getFileName2()#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                    #fName = fileName+'%05d.jpg' % (fCount+1)
                
                print('frame in gaze',framegaze)
                # textDisp = visual.TextStim(win, text='llll', wrapWidth=1800,
                #            height=70, colorSpace='rgb255',
                #            color=[255,255,255], pos = gridPoints[i])#pos=(100,0)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr

                # textDisp.draw()
                
                # win.flip()
                
                # time.sleep(0.1)
                #win.flip()
                #cv2.imwrite(calibration+'/'+fName, getFrame())#'\\'rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                
                #img=getFrame()
                #cv2.imshow('img',img)
               # cv2.imshow('frame',getFrame())#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                headerInfo.loc[fCount, 'x'] = gridPoints[i][0]
                headerInfo.loc[fCount, 'y'] = gridPoints[i][1]
                headerInfo.loc[fCount, 'dotNr'] = i
                headerInfo.loc[fCount, 'arrowOri'] = leftRight[lr]
                headerInfo.loc[fCount, 'fName'] = fName
                headerInfo.loc[fCount, 'sampTime'] = sampTime
                headerInfo.loc[fCount, 'minute'] = datetime.datetime.now().minute
                headerInfo.loc[fCount, 'seconds'] = time.time()
                print(i)
                fCount+=1
                ff=fCount#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                curFCount+=1
                    
                print("lalalalalallalalallallallallallalallallallallalallallallalalallalallalallalllalallal")
                     
                # Increase frame counters
                
                
            # Go to next dot after nFramesPerDot
            if curFCount >= nFramesPerDot:
                break

        # Check abort
        escapeKey = getKey(['escape'], waitForKey=False)
        if escapeKey[0] == 'escape':
            break
        closeKey = getKey(['ctrl'], waitForKey=False)#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        if closeKey[0] == 'ctrl':
            win.close()
        # Draw arrow and get response
        respIdxEnd = fCount-1#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        drawArrow(gridPoints[i], leftRight[lr])
        win.flip() 
        time.sleep(0.150)
        win.flip()
        resp = getKey(timeOut=1)[0]
        headerInfo.loc[respIdxStart:respIdxEnd, 'Resp'] = resp
        if leftRight[lr] == resp: 
            headerInfo.loc[respIdxStart:respIdxEnd, 'corrResp'] = True
            flagwrite=1 #rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
            drawArrow(gridPoints[i], leftRight[lr], [0,255,0])             
        else:
            headerInfo.loc[respIdxStart:respIdxEnd, 'corrResp'] = False
            drawArrow(gridPoints[i], leftRight[lr], [255,0,0])
            
        # Draw response
        win.flip()
        time.sleep(0.25)
        win.flip()
        
        # Break between blocks
        
        if (i+1)%nrPoints == 0 and i-1 != len(gridPoints):
            aastr=str(aa)
            drawText(win, 'Stand at position: '+aastr+' \n Press space to start calibration!')#j=1 j=j+1
            #drawText(win, 'Break! \n\nPress space to continue')
            aa=aa+1
            print(aa)
            win.flip()
            time.sleep(0.5)
        
    win.flip()
    closeKey = drawText(win,'quit',['ctrl'])[0]#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    if closeKey[0] == 'ctrl':#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        win.close()
    
    return headerInfo


def determineIfEye(searchForEye = 3):
    sDetectEye = time.time()
    while True:
        
        eyes, frame = getEyeFrame()
        if eyes:
            return True
            
        if (time.time()- sDetectEye) > searchForEye:
            print('No face and/or eyes found!')
            return False
        
#==============================================================================
# WebCam code
#==============================================================================
def getFrame():
    
    #return video_capture.read()[1] rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    frame=connect.get_img()
    return frame
   

def getEyeFrame():
    # Capture frame-by-frame
   # ret, frame = video_capture.read()'rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    frame=connect.get_img()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

    # Draw a rectangle around the faces
    eyes = eye_cascade.detectMultiScale(gray)
    detected = False
    if len(eyes) == 2 and len(faces)>0:
        detected = True
        
    return detected, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

def dispCalVid(loc, f, fps=33):
    data = pd.read_pickle(loc+'\\'+f)
    try:
        for i in range(len(data)):
            frame = cv2.imread(loc+'\\'+data['fName'][i])
            cv2.putText(frame,'CalDot: '+str(int(data['dotNr'][i])),(0,25), 0, 0.7, (0,0,255), 2, cv2.LINE_AA)
            cv2.resizeWindow("Calibration", 400, 300) #rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
            cv2.imshow('Calibration', frame)
            cv2.waitKey(int(1000/fps))
    except:
        pass

def getFileName(ppNr):
    participant = 'PP%03d' % ppNr
    # Check for existing files
    if not os.path.exists(participant):
        os.mkdir(participant)
    else:
        it = 1
        newDir = participant
        while os.path.exists(newDir):
            newDir = participant+'_%03d' % it
            it+=1
            if it > 100:
                break
        participant = newDir
        os.mkdir(newDir)
    return participant


def getFileName2():
    now = datetime.datetime.now()
    identifier = np.random.randint(0, 99999999)
    date = '{}_{}_{}_{}_{}_{}{:08d}'.format(datetime.datetime.now().year, datetime.datetime.now().month, 
                   datetime.datetime.now().day, datetime.datetime.now().hour, datetime.datetime.now().minute, datetime.datetime.now().second, identifier)#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    # Check for existing files
    if not os.path.exists(date):
        os.mkdir(date)
    else:
        it = 1
        newDir = date
        while os.path.exists(newDir):
            identifier = np.random.randint(0, 99999999)
            newDir = '{}_{}_{}_{}_{}_{}_{:08d}'.format(now.year, now.month, 
                           now.day, now.hour, now.minute, now.second, identifier)
            it+=1
            if it > 100:
                break
        date = newDir
        os.mkdir(date)
    return date


def showFaceAndEye():
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    #video_capture = cv2.VideoCapture(0)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
    while True:
        # Capture frame-by-frame
       # ret, frame = video_capture.read()rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
        frame=connect.get_img()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
            # only search for eyes in faces
            eyes = eye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
            for (ex,ey,ew,eh) in eyes:
                  cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(255,0,0),2)
       
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break                                                                                                                                                                                                                                                                                        
    
#==============================================================================
# # # Test code
#==============================================================================
#==============================================================================
#==============================================================================

#==============================================================================
# Settings
#==============================================================================
pc = 'CLL_mac:{}'.format(getMac())
#screenRes = (1920,1080)(3300,2100)(3840,2160)3830,2150)(1730 1100)(1728 1117)rrrrrrrrrrrrrrrrrrrrrrr rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
screenRes=(3840,2160)
#screenRes=(1500,1000)
bgColor = (0,0  ,0)
screenNr = 1
#fullScreen = True rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
fullScreen = False
# Get correct file names
#participant = getFileName2()#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
participant='cll' 
#fileName = '{}_'.format(participant)#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
fileName = getFileName2()
#==============================================================================
# Initiate webcam   
#==============================================================================
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#video_capture = cv2.VideoCapture(0)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
#connect = socket_connection(ip='10.15.3.25', port=12346, camera=5)
#connect.adjust_head(0.1, 0.0)
time.sleep(0.5)
#connect.adjust_head(-0.1, 0.0)
#==============================================================================
# Initiate psychopy                                                                                                                                                                                                                                                                                                              
#==============================================================================    
mon = monitors.Monitor('testMonitor',width=47,distance=75)
win = visual.Window(pos=(0,0),units='pix',monitor=mon,size=screenRes,colorSpace='rgb255',
                    color = bgColor, screen = screenNr, fullscr=fullScreen,
                    gammaErrorPolicy = "ignore")
mouse = event.Mouse(win=win)
mouse.setVisible(0)
print("I AM HERE")
#==============================================================================
# Run calibration
#==============================================================================
dataframe = calibration(win, fileName, participant, pc)
mouse.setVisible(2)#2rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
win.close()
flagwrite=1#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
#=============   =================================================================
# Save data 
#==============================================================================
dataframe.to_pickle(participant+'\\'+fileName+'Header.p')
#dataframe.to_csv(participant+'\\'+fileName+'Header.csv',index=False)rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
dataframe.to_csv(fileName+'Header.csv',index=False)
##==============================================================================
# Clean up
##==============================================================================
#video_capture.release() rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
cv2.destroyAllWindows()
cv2.waitKey(1)#rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
#==============================================================================
# Display video
#==============================================================================
#dispCalVid(participant,fileName+'Header.p')
#cv2.destroyAllWindows()

#dataframe=headerInfo
dataframe=headerInfo
dataframe.to_csv(fileName+'Header.csv',index=False)
















