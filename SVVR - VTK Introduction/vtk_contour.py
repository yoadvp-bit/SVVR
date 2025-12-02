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
  if len(argv) < 2:
    sys.stderr.write("Usage: %s <volume.vtk> [contourvalue]\n" % argv[0])
    return 1

  filename = argv[1]

  if not os.path.exists(filename):
    sys.stderr.write("file '%s' not found\n" % filename)
    return 1

  reader = vtk.vtkDataSetReader()
  reader.SetFileName(filename)
  reader.Update()

  data = reader.GetOutput()
  print("Dimensions   : ", data.GetDimensions())
  print("Scalar range : ", data.GetScalarRange())
  srange = data.GetScalarRange()

  if len(argv) == 3:
    value = int(argv[2])
  else:
    value = (srange[1] - srange[0])/2

  ren = vtk.vtkRenderer()
  renWin = vtk.vtkRenderWindow()
  renWin.AddRenderer(ren)
  renWin.SetSize(WIDTH, HEIGHT)

  # create a renderwindowinteractor
  iren = vtk.vtkRenderWindowInteractor()
  iren.SetRenderWindow(renWin)

  # Contour pipeline:
  contour = vtk.vtkContourFilter()
  contour.SetInputConnection(reader.GetOutputPort())
  contour.ComputeScalarsOff() ;# the default is to generate scalars
  contour.SetValue(0, value)

  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(contour.GetOutputPort())

  actor = vtk.vtkActor()
  actor.SetMapper(mapper)

  ren.AddActor(actor)

  iren.Initialize()
  iren.Start()

main(sys.argv)
