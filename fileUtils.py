# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:18:40 2018

@author: renjch
"""

import time
import datetime
import xmlUtils
import os
import glob
import os,shutil

def moveFile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile, dstfile)          #移动文件
        print ("move %s -> %s" % (srcfile, dstfile))

def copyFile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s" % ( srcfile,dstfile))

def findFiles(path, ext = '*'):
    file = os.path.join(path, ext)
    fileList = []
#    print(file)
    for f in glob.glob(file):
#        print(f)
        fileList.append(f)
    return fileList

def names(path):
    '''
    return the dirname, filename, name, ext of input path
    '''
    [dirname, filename] = os.path.split(path)
    [name, ext] = os.path.splitext(filename)
    return dirname, filename, name, ext

def fileName(path):
    return names(path)[2]

def fileFullName(path):
    [dirname, filename] = os.path.split(path)
    return filename

def findPairFile(file, findPath, pairExt = ''):
    dirname, filename, name, ext = names(file)
    if pairExt == '':
        pairExt = ext
    findfile = os.path.join(findPath, name + pairExt)
    return findfile, os.path.exists(findfile)



def delFiles(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            delFiles(c_path)
        else:
            os.remove(c_path)

def ensurePathExist(path, clear = False):
    if (os.path.exists(path) == False):
        os.makedirs(path)
    else:
        if (clear):
            delFiles(path)

def ensurePathExistAndEmpty(path):
    ensurePathExist(path, True)  
    
    
if __name__ == '__main__':
    try:
        copyFile('H:/VOC2018/Annotations/00000002.xml', 'H:/VOC2018/Annotations/00000002.xml')
    except shutil.SameFileError:
        pass #pass error samefileerror
    print('here')