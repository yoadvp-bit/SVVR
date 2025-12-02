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

class vtkTimerCallback():
    def __init__(self, steps, actor, iren):
        self.timer_count = 0
        self.steps = steps
        self.actor = actor
        self.iren = iren
        self.timerId = None

    def execute(self, obj, event):
        step = 0
        while step < self.steps:
            print(self.timer_count)
            #self.actor.SetPosition(self.timer_count / 100.0, self.timer_count / 100.0, 0)
            self.actor.RotateX(1)
            iren = obj
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timerId:
            iren.DestroyTimer(self.timerId)

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

  #for i in range(1,360):
  #  actor.RotateX(1)
  #  renWin.Render()
  #  print(i)

  iren.Initialize()

  # Sign up to receive TimerEvent
  cb = vtkTimerCallback(360, actor, iren)
  iren.AddObserver('TimerEvent', cb.execute)
  cb.timerId = iren.CreateRepeatingTimer(500)

  iren.Start()

main(sys.argv)
