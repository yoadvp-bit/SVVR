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
  cone.SetResolution(15) ;# default resolution

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

  # Create a slider to set the cone's resolution.
  # First, the representation of the slider:
  slider_rep = vtk.vtkSliderRepresentation2D()
  slider_rep.SetMinimumValue(1)
  slider_rep.SetMaximumValue(100)
  slider_rep.SetValue(15) ;# default value, same as set above
  slider_rep.SetTitleText("Resolution")
  slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
  slider_rep.GetPoint1Coordinate().SetValue(0.3, 0.05)
  slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
  slider_rep.GetPoint2Coordinate().SetValue(0.7, 0.05)
  slider_rep.SetSliderLength(0.02)
  slider_rep.SetSliderWidth(0.03)
  slider_rep.SetEndCapLength(0.01)
  slider_rep.SetEndCapWidth(0.03)
  slider_rep.SetTubeWidth(0.005)
  slider_rep.SetLabelFormat("%3.0lf")
  slider_rep.SetTitleHeight(0.02)
  slider_rep.SetLabelHeight(0.02)

  # The slider (see https://vtk.org/doc/nightly/html/classvtkSliderWidget.html):
  slider = vtk.vtkSliderWidget()
  slider.SetInteractor(iren)
  slider.SetRepresentation(slider_rep)
  slider.KeyPressActivationOff()
  slider.SetAnimationModeToAnimate()
  slider.SetEnabled(True)

  # Define what to do if the slider value changed:
  def processEndInteractionEvent(obj,event):
    value = int (obj.GetRepresentation().GetValue())
    cone.SetResolution(value)

  #slider.AddObserver("EndInteractionEvent", processEndInteractionEvent) ;# change value only when released
  slider.AddObserver("InteractionEvent", processEndInteractionEvent) ;# change value when moved

  iren.Initialize()
  iren.Start()

main(sys.argv)
