# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:29:50 2018

@author: renjch
"""

import numpy as np
import cv2
from lxml import etree
import datetime
import sys
import getopt
import random

from optparse import OptionParser 

#'''
#
#<annotation verified="yes">
#	<folder>截屏</folder>
#	<filename>图像 001.png</filename>
#	<path>C:\Users\renjch\Pictures\截屏\图像 001.png</path>
#	<source>
#		<database>Unknown</database>
#	</source>
#	<size>
#		<width>1920</width>
#		<height>1080</height>
#		<depth>3</depth>
#	</size>
#	<segmented>0</segmented>
#</annotation>
#
#
#'''

import os
from lxml import etree, objectify

import time

import xmlUtils
import fileUtils
import imgUtils
import glob


        
def makeNameByTimestampMs():
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H%M%S')
    ms_time = lambda:int(round(time.time() * 1000) % 1000)
    return time_str + str(ms_time()).zfill(3)  

# move Annotation & image
# move image to JPEGImages and move xml to Annotation
# and check names, and make image set file
# this function maybe useful for after label image by labelImage 
    
def moveToVOCDataSet(imageFolder, xmlFolder, vocFolder):
    #just move without check
    pass

def renameAnnotationFolder(path, outFolder, newName):
    file = os.path.join(path, "*.xml")
    for f in glob.glob(file):
        tree = xmlUtils.read_xml(f)  
        [dirname, filename] = os.path.split(f)
        nodes = xmlUtils.find_nodes(tree, "folder")  
        xmlUtils.change_node_text(nodes, newName)  
        xmlUtils.write_xml(tree, os.path.join(outFolder, filename))  
        

def genBackgroundAnnotation(path, outpath):
    
    if (os.path.exists(outpath) == False):
        os.makedirs(outpath)
    
    if (os.path.exists(path) == False):
        print('Image can not find!\n');
        return
    
    img = cv2.imread(path)
    [dirname, filename] = os.path.split(path)
    [name, ext] = os.path.splitext(filename)
    [dirname, n] = os.path.split(dirname)
    dirname = os.path.basename(dirname) # xxx/VOC2018/JPEGImages 要获取 VOC2018那个目录
    print(name, ext)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder(dirname),
        E.filename(filename),
        E.path(path),
        E.source(
            E.database('Unknown'),
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0),
    )
    xmlPath = os.path.join(outpath, name + '.xml')
    print('Generate: ' + xmlPath + '\n')
    etree.ElementTree(anno_tree).write(xmlPath, pretty_print=True)
    
def genBackgroundAnnotation2(path, outpath):
    
    if (os.path.exists(outpath) == False):
        os.makedirs(outpath)
    
    if (os.path.exists(path) == False):
        print('Image can not find!\n');
        return
    
    img = cv2.imread(path)
    [dirname, filename] = os.path.split(path)
    [name, ext] = os.path.splitext(filename)
    [dirname, n] = os.path.split(dirname)
    dirname = os.path.basename(dirname) # xxx/VOC2018/JPEGImages 要获取 VOC2018那个目录
    print(name, ext)
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder(dirname),
        E.filename(filename),
        E.path(path),
        E.source(
            E.database('Unknown'),
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0),
    )
    xmlPath = os.path.join(outpath, name + '.xml')
    print('Generate: ' + xmlPath + '\n')
    etree.ElementTree(anno_tree).write(xmlPath, pretty_print=True)    

def exportNegativeVideo(videoPath, vocPath, skipFrame = 15, show = False):
    if (os.path.exists(videoPath) == False):
        print('Can not find video file.')
        return
    savePath = os.path.join(vocPath, 'JPEGImages')
    xmlOutPath = os.path.join(vocPath, 'Annotations')
    imgSetPath = os.path.join(vocPath, 'ImageSets' + os.sep + 'Main')
    imgList = []
    cap = cv2.VideoCapture(videoPath)
    
    while True:
        ret,frame = cap.read()
        if (ret == False):
            break
    
        imageName = makeNameByTimestampMs()
        imagePath = os.path.join(savePath, imageName + '.jpg')
        
        cv2.imwrite(imagePath, frame)
        imgList.append(imageName)
        genBackgroundAnnotation(imagePath, xmlOutPath)
        
        for i in range(skipFrame):
            ret,frame = cap.read()
            if (ret == False):
                break
        
        if (show):
            cv2.imshow('', frame)
            cv2.waitKey(30) 
    
    imgSetFile = os.path.join(imgSetPath, makeNameByTimestampMs() + '.txt')
    print('ImageSet save: ' + imgSetFile)
    fp = open(imgSetFile, 'w+')
    for line in imgList:
        fp.write(line + ' -1\n')
    fp.close()
    
    cap.release()#释放摄像头，调用
    if (show):
        cv2.destroyAllWindows()#关闭所有图像窗口。
    
    return 

def genSampleJointBackground(img, sampleFile, annotationFile, vocPath):
    pass

def export_background_to_voc(samplePath, annotationPath, backgroundPath, vocPath, show = False):
        
    backgroundFileList = fileUtils.findFiles(backgroundPath, '*.jpg')
    
    sampleFileList = fileUtils.findFiles(samplePath, '*.jpg')
    sampleCnt = len(sampleFileList)
    
    JPEGImages = os.path.join(vocPath, 'JPEGImages')
    Annotations = os.path.join(vocPath, 'Annotations')
    
    year = datetime.datetime.now().year
    
    genSampleList = []
    
    for backgroundFile in backgroundFileList:
        sampleFile = ''
        sampleFileName = ''
        annotationFile = ''
        backgroundFileName = fileUtils.fileName(backgroundFile)
        while True:
            randIdx = random.randrange(0, sampleCnt)
            sampleFile = sampleFileList[randIdx]
            sampleFileName = fileUtils.fileName(sampleFile)
            annotationFile = os.path.join(annotationPath, sampleFileName + '.xml')
            if os.path.exists(annotationFile):
                break
        # Year_Joint_Foreground_background.jpg / .xml    
        jointSampleName = str(year) + '_Joint_' + sampleFileName + '_' + backgroundFileName + '.jpg'
        jointXmlName = str(year) + '_Joint_' + sampleFileName + '_' + backgroundFileName + '.xml'
        img = imgUtils.joint_foreground_background(sampleFile, backgroundFile)
        
        if img is not None:
            attrDict = {}
            attrDict['filename'] = jointSampleName
            attrDict['size/width'] = str(img.shape[1])
            attrDict['size/height'] = str(img.shape[0])
            xmlUtils.change_allnode_text_by_name2(annotationFile, os.path.join(Annotations, jointXmlName), attrDict)
            cv2.imwrite(os.path.join(JPEGImages, jointSampleName), img)
            genSampleList.append(os.path.join(JPEGImages, jointSampleName))
            if show:
                cv2.imshow('', img)
                cv2.waitKey(1)

    if show:
        cv2.destroyAllWindows()
        
    return genSampleList
    

def makeVOCDataSet(rootPath, year, key = '', new = False):
    #make Annotations
    rootPath = os.path.join(rootPath, 'VOC' + year + key)
    fileUtils.ensurePathExist(rootPath, new)
    
    dirs = ['Annotations', 'ImageSets', 'JPEGImages', 'SegmentationClass', 'SegmentationObject']
    subDirs = ['ImageSets' + os.sep + 'Action',
               'ImageSets' + os.sep + 'Layout',
               'ImageSets' + os.sep + 'Main',
               'ImageSets' + os.sep + 'Segmentation']
    for d in dirs:
        path = os.path.join(rootPath, d)
        fileUtils.ensurePathExist(path, new)
    
    for sd in subDirs:
        path = os.path.join(rootPath, sd)
        fileUtils.ensurePathExist(path, new)
    
    return rootPath
    
#path = makeVOCDataSet("H:\\test\\test", '2018')
#exportNegativeVideo('background.avi', path)
#renameAnnotationFolder(r"G:\label\box", r"H:\x", 'VOC2018')
def export_labelimage_to_voc(samplePath, annotationPath, vocPath):
    sampleList = fileUtils.findFiles(samplePath, '*.jpg')
    genSampleList = []
    for file in sampleList:
        # 获取目标文件名
        xml, exist = fileUtils.findPairFile(file, annotationPath, '.xml')
        xmlSavePath, _ = fileUtils.findPairFile(xml, os.path.join(vocPath, 'Annotations'))
        jpgSavePath, _ = fileUtils.findPairFile(file, os.path.join(vocPath, 'JPEGImages'))
        vocName = fileUtils.fileName(vocPath)
        attrDict = {}
        attrDict['folder'] = vocName
        attrDict['path'] = jpgSavePath
        print(xml, exist, xmlSavePath, jpgSavePath)
        if exist:
            # save annotation xml to dest path
            xmlUtils.change_allnode_text_by_name2(xml, xmlSavePath, attrDict)
            # save jpg image sample to dest path
            fileUtils.copyFile(file, jpgSavePath)
            genSampleList.append(jpgSavePath)
    
    # return all sample list, than save to the .data set list.
    return genSampleList

def action_create(path, label, del_if_exist):
    year = datetime.datetime.now().year
    makeVOCDataSet(path, year, '', del_if_exist)

def action_export(samplePath, annotationPath, backgroundPath, vocPath):
    sampleList = []
    list1 = export_labelimage_to_voc(samplePath, annotationPath, vocPath)
    sampleList.extend(list1)
    if os.path.exists(backgroundPath):
        list2 = export_background_to_voc(samplePath, annotationPath, backgroundPath, vocPath, show = True)
        sampleList.extend(list2)
        
    # save image sample list to .data set list.
    if len(sampleList) > 0:
        name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fullname = os.path.join(vocPath, 'ImageSets', 'Main', name + '.txt')
        
        with open(fullname, 'w+') as f:
            for sample in sampleList:
                name = fileUtils.fileName(sample)
                f.write(name + ' ' + '1\n')
                                                                                                                        
parser = OptionParser(usage="%prog [options]")
parser.add_option("--action",action="store",type="string",dest="action",help="operate")
parser.add_option("-d","--data_dir",action="store",default='./',dest="data_dir",help="dir to save the .data set")
parser.add_option("--label", action="store", default='',dest="label",help=".data set lable")
parser.add_option("--del_if_exist",action="store", default=False, dest="del_if_exist",help="del_if_exist")

parser.add_option("--sample_path", action="store", dest="sample_path", help="the machine to be check")
parser.add_option("--annotation_path", action="store", dest="annotation_path", help="the machine to be check")
parser.add_option("--background_path", action="store", dest="background_path", default='', help="the machine to be check")
parser.add_option("--voc_path", action="store", dest="sample_path", help="the machine to be check")

try:
    (options,args)=parser.parse_args()   
    if options.action:
        action = options.action
        if action == 'create':
            action_create(options.data_dir, options.label, options.del_if_exist)
        elif action == 'export':
            action_export()
    else:
        pass
except:
    pass

ACTION = 'create'
DATA_DIR = 'C:/'
KEY = 'bands'
DEL_IF_EXIST = False
samplePath = 'C:/samples/band'
annotationPath = 'C:/samples/band_annotations'
backgroundPath = ''

if __name__ == '__main__':

    year = datetime.datetime.now().year
    
    vocPath = DATA_DIR
    #print(DEL_IF_EXIST)
    if ACTION == 'create':
        vocPath = makeVOCDataSet(DATA_DIR, str(year), KEY, DEL_IF_EXIST)
    
    #vocName = fileUtils.fileName(pathdest)
    #print(vocName)
    
    action_export(samplePath, annotationPath, backgroundPath, vocPath)
#
#def TestGetOpt():
#  day = 1
#  files = 1
#  try:
#    opts, args = getopt.getopt(sys.argv[1:],'d:f:h',['days=','files=','help'])
#  except getopt.GetoptError:
##     usage()
#     sys.exit()
#
#  print (opts)
#  print (args)
#  for o, a in opts:
#     if o in ("-h", "--help"):
##         usage()
#         sys.exit()
#     elif o in ("-d", "--days"):
#         day = a
#     elif o in ("-f", "--files"):
#         files = a
#  print (day)
#  print (files)
#  
#TestGetOpt()  



