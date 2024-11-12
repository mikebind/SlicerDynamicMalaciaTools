import logging
import os
from typing import Annotated, Optional, List

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLTableNode
import numpy as np

#
# HannahCanCode
#


class HannahCanCode(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _(
            "HannahCanCode"
        )  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#HannahCanCode">module documentation</a>.
"""
        )
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(
            """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""
        )


#
# HannahCanCodeParameterNode
#


@parameterNodeWrapper
class HannahCanCodeParameterNode:
    """
    The parameters needed by module.

    inputTableNode - The table to look for minimum CSA in
    minimumCSA - The minimum cross-sectional area in the table
    minimumCSAFrameNumber - The frame number
    """

    inputTableNode: vtkMRMLTableNode
    minimumCSA: float
    minimumCSAFrameNumber: int


#
# HannahCanCodeWidget
#


class HannahCanCodeWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/HannahCanCode.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = HannahCanCodeLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Buttons
        self.ui.findMinimumCSAButton.connect(
            "clicked(bool)", self.onFindMinimumCSAButton
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputTableNode:
            firstTableNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLTableNode")
            if firstTableNode:
                self._parameterNode.inputTableNode = inputTableNode

    def setParameterNode(
        self, inputParameterNode: Optional[HannahCanCodeParameterNode]
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputTableNode:
            self.ui.findMinimumCSAButton.toolTip = _(
                "Find the minimum CSA and frame containing it!"
            )
            self.ui.findMinimumCSAButton.enabled = True
        else:
            self.ui.findMinimumCSAButton.toolTip = _(
                "Select input table to enable button"
            )
            self.ui.findMinimumCSAButton.enabled = False

    def onFindMinimumCSAButton(self) -> None:
        """This is run when the button is clicked.  It should run a function from
        the logic to find the minimum cross-sectional area across all frames, and
        return the value of that minimum, as well as which frame number it appears in.
        """
        tableNode = self._parameterNode.inputTableNode
        minimumCSA, frameNumber = self.logic.getMinimumCSAandFrameNumber(tableNode)
        # Update the text of the UI to show the results
        self.ui.minimumCSArea.text = f"{minimumCSA} cm^2"
        self.ui.minimumCSAFrameNum.text = f"{frameNumber}"
        # Update the parameter node to store the results
        self._parameterNode.minimumCSA = minimumCSA
        self._parameterNode.minimumCSAFrameNumber = frameNumber


#
# HannahCanCodeLogic
#


class HannahCanCodeLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return HannahCanCodeParameterNode(super().getParameterNode())

    def getMinimumCSAandFrameNumber(self, tableNode):
        """This function does the work of finding which
        frame contains the minimum CSA and the minimum
        CSA value. It is called from onFindMinimumCSAButton(), which
        is called when the user clicks on the button.
        """
        # NOTE: HANNAH'S CODE GOES HERE
        # It should find the minimum CSA in the given input tableNode,
        # (which should be a merged table with the CSA for each frame
        # in columns). The frame number of the frame with the minimum
        # CSA in it should be put into a variable called
        # frameNumberWithMinimumCSA, and the minimum CS area value
        # itself should be put into a variable called minimumCSA.
        # Then, the "return" line below, will send those values back
        # to the function which called this one.

        # These are the right answers for the first test, but will
        # fail the second test (when running using the "Reload and Test"
        # button), and obviously isn't a correct solution in general.
        minimumCSA = 3
        frameNumberWithMinimumCSA = 1

        return minimumCSA, frameNumberWithMinimumCSA


#
# HannahCanCodeTest
#


class HannahCanCodeTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_HannahCanCode1()
        self.test_HannahCanCode2()

    def test_HannahCanCode1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting Test 1")

        tableRows1 = np.array([[10, 20, 10], [15, 14, 19], [4, 3, 6], [9, 9, 9]])
        columnNames = [f"CSA (#{frameIdx})" for frameIdx in range(3)]
        correctMinCSA = 3
        correctFrameNum = 1

        tableNode1 = self.makeTableNodeFromNumpyArray(
            tableRows1, "Test1_Table", columnNames
        )

        logic = HannahCanCodeLogic()
        outputMinCSA, outputFrameNum = logic.getMinimumCSAandFrameNumber(tableNode1)

        if self.checkAnswers(
            outputMinCSA, correctMinCSA, outputFrameNum, correctFrameNum
        ):
            self.delayDisplay("Test 1 passed!")
        else:
            self.delayDisplay("Test 1 failed!")

    def test_HannahCanCode2(self):
        """A more complex test which makes sure that the RAS coordinate columns are not
        being treated as CSAs.
        """
        self.delayDisplay("Starting Test 2")
        tableRows2 = np.array(
            [
                [10, 20, 10, -1, 90, -25.1],
                [15, 14, 19, 4, 86, -20],
                [4, 3, 2.2, -3, 88, -15],
                [9, 9, 9, 0, 87, -10],
            ]
        )
        columnNames = [f"CSA (#{frameIdx})" for frameIdx in range(3)]
        columnNames.extend(["R_coord", "A_coord", "S_coord"])
        correctMinCSA = 2.2
        correctFrameNum = 2
        tableNode2 = self.makeTableNodeFromNumpyArray(
            tableRows2, "Test2_Table", columnNames
        )
        logic = HannahCanCodeLogic()
        outputMinCSA, outputFrameNum = logic.getMinimumCSAandFrameNumber(tableNode2)
        if self.checkAnswers(
            outputMinCSA, correctMinCSA, outputFrameNum, correctFrameNum
        ):
            self.delayDisplay("Test 2 passed, great job!")
        else:
            self.delayDisplay("Test 2 failed!")
            # Check if the minCSA is the min coord value, if so give additional guidance
            if outputMinCSA == -25.1:
                self.delayDisplay(
                    "Output minimum CSA is equal to the minimum coordinate value! Make sure you don't include coordinates in your minimum calculations!"
                )

    def checkAnswers(
        self, outputMinCSA, correctMinCSA, outputFrameNum, correctFrameNum
    ):
        """Check whether the outputs match the correct answers.  If not, show an
        warning message with information about the mismatch.
        """
        if outputMinCSA == correctMinCSA:
            # Found the correct minimum CSA!
            csaCorrect = True
        else:
            slicer.util.warningDisplay(
                f"The correct minimum CSA was {correctMinCSA}, but the function returned {outputMinCSA}!"
            )
            csaCorrect = False
        if outputFrameNum == correctFrameNum:
            # Found the correct frame number
            frameCorrect = True
        else:
            frameCorrect = False
            slicer.util.warningDisplay(
                f"The correct minimum Frame Number was {correctFrameNum}, but the function returned {outputFrameNum}!"
            )
        # Return true if the test passed, false, if the test failed
        if csaCorrect and frameCorrect:
            return True
        else:
            return False

    def makeTableNodeFromNumpyArray(
        self,
        numpyTable: np.ndarray,
        nodeName: str = "TestTable",
        columnNames: Optional[List[str]] = None,
    ) -> vtkMRMLTableNode:
        """Creates a tableNode from a 2D numpy array. Optionally name the columns and
        set the tableNode name. Returns the created tableNode.
        """
        # Create a vtkMRMLTableNode
        tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", nodeName)
        # Get the vtkTable from the MRML node
        vTable = tableNode.GetTable()
        # Fill in the table data
        for colIdx in range(numpyTable.shape[1]):
            column = vtk.vtkFloatArray()
            # If column names are provided, apply them
            if columnNames:
                column.SetName(columnNames[colIdx])
            # Loop over the rows, inserting each row's value for this column
            for rowIdx in range(numpyTable.shape[0]):
                column.InsertNextValue(numpyTable[rowIdx, colIdx])
            # Add the column to the vtkTable
            vTable.AddColumn(column)
        # Now that the table is filled in, we can return the containing node for use elsewhere
        return tableNode
