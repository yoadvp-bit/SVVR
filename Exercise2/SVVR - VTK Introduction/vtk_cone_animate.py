#!/usr/bin/env vtkpython

import sys
import os
import vtk

# Window width and height
WIDTH=800
HEIGHT=800

v = vtk.vtkVersion()
version = v.GetVTKSourceVersion()
print(version)

def main(argv):

  cone = vtk.vtkConeSource()
  cone.SetResolution(15)

  ren = vtk.vtkRenderer()
  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)
  renWin.SetSize(WIDTH, HEIGHT)

  # create a renderwindowinteractor
  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renWin)

  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(cone.GetOutputPort())

  actor = vtk.vtkActor()
  actor.SetMapper(mapper)

  ren.AddActor(actor)

  for i in range(1,360):
    actor.RotateX(1)
    renWin.Render()
    print(i)

  iren.Initialize()
  iren.Start()

main(sys.argv)
