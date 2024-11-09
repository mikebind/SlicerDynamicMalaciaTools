import logging
import os
from typing import Annotated, Optional, List, Tuple, Dict

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

import numpy as np
import re
import pathlib
from vtk.util import numpy_support

from slicer import (
    vtkMRMLScalarVolumeNode,
    vtkMRMLSegmentationNode,
    vtkMRMLMarkupsCurveNode,
    vtkMRMLSequenceNode,
    vtkMRMLSequenceBrowserNode,
    vtkMRMLTableNode,
)


SEGMENT_NAME_DIVIDER = "--Only Some Frames--"
#
# DynamicMalaciaTools
#


class DynamicMalaciaTools(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Dynamic Malacia Tools")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Airway")]
        self.parent.dependencies = [
            "CrossSectionAnalysis"
        ]  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "Mike Bindschadler (Seattle Children's Hospital)"
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """This module provides some tools for analyzing tracheal malacia in 4D-CT imaging.
See more information in <a href="https://github.com/mikebind/SlicerDynamicMalaciaTools">module repository</a>.
"""
        )
        self.parent.acknowledgementText = _(
            """
This file was originally developed by Mike Bindschadler as part of work at Seattle Children's Hospital.
"""
        )
        # Skip sample data registration for now (until we have a sample data set we want to use)
        ### # Additional initialization step after application startup is complete
        ### slicer.app.connect("startupCompleted()", registerSampleData)


#
# DynamicMalaciaToolsParameterNode
#


@parameterNodeWrapper
class DynamicMalaciaToolsParameterNode:
    """
    The parameters needed by module.

    """

    centerlineNode: vtkMRMLMarkupsCurveNode
    segmentationSequence: Optional[vtkMRMLSequenceNode] = None
    segmentName: Optional[str] = None
    segmentFrames: Optional[List[int]] = None
    outputTableSequence: Optional[vtkMRMLSequenceNode] = None
    mergedCSATableNode: Optional[vtkMRMLTableNode]

    volumeLimitingSegmentationNode: Optional[vtkMRMLSegmentationNode] = None
    volumeLimitingSegmentID: str

    volumeOutputTableNode: Optional[vtkMRMLTableNode]

    saveDirectory: pathlib.Path


#
# DynamicMalaciaToolsWidget
#


class DynamicMalaciaToolsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/DynamicMalaciaTools.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = DynamicMalaciaToolsLogic()

        # For the sequence node selectors, we want to restrict them so that
        # they only show sequence nodes which hold the correct data node type
        # Thankfully, this is already set as a MRML attribute, and we can filter
        # by it!
        self.ui.segmentationSequenceSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLSegmentationNode"
        )
        self.ui.outputTableSequenceNodeSelector.addAttribute(
            "vtkMRMLTableNode", "DataNodeClassName", "vtkMRMLTableNode"
        )

        # Connections
        # qMRMLSegmentSelectorWidgets cannot yet be handled via the parameter node
        # wrapper, so must still be handled manually here
        self.ui.volumeLimitingSegmentationSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.onVolumeLimitingSegmentationSelectionChange,
        )
        self.ui.volumeLimitingSegmentationSelector.connect(
            "currentSegmentChanged(QString)",
            self.onVolumeLimitingSegmentSelectionChange,
        )
        self.ui.segmentationSequenceSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.onSegmentationSequenceSelectorChange,
        )
        self.ui.segmentNameComboBox.connect(
            "currentTextChanged(QString)", self.onSegmentNameChanged
        )
        self.ui.segmentNameComboBox.connect(
            "activated(int)", self.onSegmentNameActivated
        )

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Buttons
        self.ui.runCSAAnalysisButton.connect("clicked(bool)", self.onRunCSAButtonClick)
        self.ui.findLimitedVolumeButton.connect(
            "clicked(bool)", self.onFindLimitedVolumesButtonClick
        )
        self.ui.saveOutputTablesButton.connect(
            "clicked(bool)", self.onSaveOutputTablesButtonClick
        )
        self.ui.makeCSAPlotButton.connect(
            "clicked(bool)", self.onMakeCSAPlotButtonClick
        )

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        # Make sure segment list is initialized if there is an
        # initial segmentation sequence
        self.onSegmentationSequenceSelectorChange(
            self._parameterNode.segmentationSequence
        )

    def onSegmentNameChanged(self, txt):
        """In the multiframe section, triggered whenever the text in the segment
        name selection combobox changes"""
        segmentName, segmentFrames = (
            self.logic.processSegmentSelectorTextToNameAndFrames(txt)
        )
        self._parameterNode.segmentName = segmentName
        self._parameterNode.segmentFrames = segmentFrames

    def onSegmentNameActivated(self, num):
        """In the multiframe section, triggered whenever a choice is made in the
        segment name selection combobox. I'm including this because I want a way
        to trigger updating even if nothing supposedly changes.
        """
        txt = self.ui.segmentNameComboBox.currentText
        self.onSegmentNameChanged(txt)

    def onRunCSAButtonClick(self):
        """Run cross-sectional analysis for all frames."""
        pn = self._parameterNode
        centerlineNode = pn.centerlineNode
        segSeq = pn.segmentationSequence
        segmentName = pn.segmentName
        segmentFrames = pn.segmentFrames
        outputTableSequence = pn.outputTableSequence
        if outputTableSequence is None:
            outputTableSequence = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSequenceNode", "CSA_TableSeq"
            )
            pn.outputTableSequence = outputTableSequence
        # We should start with a clean slate even if there were prior tables
        outputTableSequence.RemoveAllDataNodes()
        self.logic.setCompatibleSeqIndexing(outputTableSequence, segSeq)

        #
        browserNode = (
            slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(segSeq)
        )
        nFrames = segSeq.GetNumberOfDataNodes()
        if segmentFrames is None:
            # all frames
            frameIdxs = range(nFrames)
        else:
            # only certain frames
            frameIdxs = [*segmentFrames]
            slicer.util.warningDisplay(
                "Sorry, this module currently only works for segments which are present in all frames!"
            )
            return
        # Loop over frames
        for frameIdx in frameIdxs:
            browserNode.SetSelectedItemNumber(frameIdx)
            lumenNode = browserNode.GetProxyNode(segSeq)
            tempOutputTableNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTableNode", f"CSATableForFrame{frameIdx}"
            )
            self.logic.runCrossSectionalAnalysis(
                centerlineNode, lumenNode, segmentName, tempOutputTableNode
            )
            indexValue = segSeq.GetNthIndexValue(frameIdx)
            outputTableSequence.SetDataNodeAtValue(tempOutputTableNode, indexValue)
            # Remove temp copy
            slicer.mrmlScene.RemoveNode(tempOutputTableNode)
            slicer.app.processEvents()
        # Ensure outputTableSequence is linked to the browser node
        if not browserNode.IsSynchronizedSequenceNode(outputTableSequence):
            browserNode.AddSynchronizedSequenceNode(outputTableSequence)

    def onFindLimitedVolumesButtonClick(self):
        """Run the volume limiting by region across all frames, and
        report the limited volumes in a table.
        """
        pn = self._parameterNode
        limitingSegmentationNode = pn.volumeLimitingSegmentationNode
        limitingSegmentID = pn.volumeLimitingSegmentID
        segSeq = pn.segmentationSequence
        segmentName = pn.segmentName
        volumeOutputTableNode = pn.volumeOutputTableNode
        if volumeOutputTableNode is None:
            volumeOutputTableNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLTableNode", "LimitedSegmentVolumes"
            )
            pn.volumeOutputTableNode = volumeOutputTableNode
        browserNode = (
            slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(segSeq)
        )
        nFrames = browserNode.GetNumberOfItems()
        limitedSegmentVolumes = []
        for frameIdx in range(nFrames):
            browserNode.SetSelectedItemNumber(frameIdx)
            segmentationNode = browserNode.GetProxyNode(segSeq)
            limitedSegmentVolume = self.logic.getLimitedSegmentVolume(
                segmentName,
                segmentationNode,
                limitingSegmentID,
                limitingSegmentationNode,
            )
            limitedSegmentVolumes.append(limitedSegmentVolume)
        self.logic.addVolumesToTable(limitedSegmentVolumes, volumeOutputTableNode)

    def onSaveOutputTablesButtonClick(self):
        """Save available and selected output tables to spreadsheet files."""
        pn = self._parameterNode
        volumeOutputTableNode = pn.volumeOutputTableNode
        saveDir = pn.saveDirectory
        if volumeOutputTableNode:
            self.logic.saveVolumeOutputTableNodeToFile(volumeOutputTableNode, saveDir)
        if pn.outputTableSequence:
            pn.mergedCSATableNode = self.logic.consolidateCSATables(
                pn.outputTableSequence
            )
            self.logic.saveMergedCsaTableNodeToFile(pn.mergedCSATableNode, saveDir)

    def onMakeCSAPlotButtonClick(self):
        """Make a plot of CSA profiles including envelope (max/min) values and
        a line which moves with the browser frame
        """
        pn = self._parameterNode
        csaSeqNode = pn.outputTableSequence
        if csaSeqNode is None:
            slicer.util.warningDialog(
                "No CSA Table Sequence Selected; can't make plot!"
            )
            return
        mergedTableNode = pn.mergedCSATableNode
        if mergedTableNode is None:
            mergedTableNode = self.logic.consolidateCSATables(csaSeqNode)
            pn.mergedCSATableNode = mergedTableNode
        chartNode, (lowSer, highSer, dynSer) = self.logic.makeDynamicPlotChart(
            mergedTableNode=mergedTableNode, csaSeqNode=csaSeqNode
        )
        # Show this in the application
        showPlotLayoutNum = 36
        slicer.app.layoutManager().setLayout(showPlotLayoutNum)
        plotViewNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLPlotViewNode")
        plotViewNode.SetPlotChartNodeID(chartNode.GetID())

    def onSegmentationSequenceSelectorChange(self, newNode):
        """User changed selection of segmentation sequence node.  The
        list of options on the segment selection list must be updated
        to match those avaiilable in the newly selected sequence
        """
        print(f"New segSeq is named {newNode.GetName() if newNode else 'NONE'}")
        self._parameterNode.segmentationSequence = newNode
        self.updateSegmentListSelector(newNode)

    def updateSegmentListSelector(self, newSegmentationSequenceNode):
        """Update the segment selector list based on the supplied
        segmentation sequence node.  If empty, clear the options."""
        if newSegmentationSequenceNode is None:
            self.ui.segmentNameComboBox.clear()
            return
        segmentsList, partialSegmentDict = self.logic.getSegmentInfoFromSequence(
            newSegmentationSequenceNode
        )
        # Build the selector list of options from the outputs
        optionStrings = self.logic.buildOptionStrings(segmentsList, partialSegmentDict)
        # Update the segment chooser combobox widget
        self.logic.updateComboBoxOptions(self.ui.segmentNameComboBox, optionStrings)

    def onVolumeLimitingSegmentationSelectionChange(self, newSegNode):
        """Triggered when new segmentation node is selected on the
        volume limiting segmentation selection widget.
        """
        self._parameterNode.volumeLimitingSegmentationNode = newSegNode

    def onVolumeLimitingSegmentSelectionChange(self, newSegmentID):
        """Triggered when new segment is selected on the volume
        limiting segment selector widget.
        """
        self._parameterNode.volumeLimitingSegmentID = newSegmentID

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
        # if not self._parameterNode.inputVolume:
        #    firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
        #        "vtkMRMLScalarVolumeNode"
        #    )
        #    if firstVolumeNode:
        #        self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(
        self, inputParameterNode: Optional[DynamicMalaciaToolsParameterNode]
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
        """Enable or disable buttons based on whether their required inputs
        are present
        """
        pn = self._parameterNode
        # Run CSA Analysis Button
        if pn and pn.centerlineNode and pn.segmentationSequence and pn.segmentName:
            self.ui.runCSAAnalysisButton.toolTip = _(
                "Run cross-section analysis across all frames"
            )
            self.ui.runCSAAnalysisButton.enabled = True
        else:
            self.ui.runCSAAnalysisButton.toolTip = _(
                "Select centerline, segmentation sequence, and segment name to enable"
            )
            self.ui.runCSAAnalysisButton.enabled = False
        # Find Limited Volumes Button
        if pn and pn.volumeLimitingSegmentationNode and pn.volumeLimitingSegmentID:
            self.ui.findLimitedVolumeButton.enabled = True
            self.ui.findLimitedVolumeButton.toolTip = _(
                "Using the selected limiting segment, find the volume of the selected sequence segment which is within the limiting segment, in each frame"
            )
        else:
            self.ui.findLimitedVolumeButton.enabled = False
            self.ui.findLimitedVolumeButton.toolTip = _(
                "A limiting segment must be chosen before you can find limited volumes!"
            )
        # Save Output Tables To File
        if pn and pn.saveDirectory != pathlib.Path():
            # User has set the path somehow, it's OK to enable the button
            self.ui.saveOutputTablesButton.enabled = True
            self.ui.saveOutputTablesButton.toolTip = _(
                "Save tables to CSV files in the specified directory"
            )
        else:
            self.ui.saveOutputTablesButton.enabled = False
            self.ui.saveOutputTablesButton.toolTip = _(
                "Specify a save location to enable saving!"
            )
        # Make CSA Plot
        if pn and pn.outputTableSequence:
            self.ui.makeCSAPlotButton.enabled = True
            self.ui.makeCSAPlotButton.toolTip = _(
                "Make a plot of the CSA profile which varies with the sequence frame"
            )
        else:
            self.ui.makeCSAPlotButton.enabled = False
            self.ui.makeCSAPlotButton.toolTip = _(
                "Run CSA analysis to enable plotting of results!"
            )


#
# DynamicMalaciaToolsLogic
#


class DynamicMalaciaToolsLogic(ScriptedLoadableModuleLogic):
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
        return DynamicMalaciaToolsParameterNode(super().getParameterNode())

    def makeDynamicPlotChart(self, mergedTableNode, csaSeqNode):
        """This function should create a plotChart node and three
        plotSeries nodes showing the envelope of CSA values (2 of the
        3 series) and one profile with the current CSA (which should
        vary with the sequence browser index).
        """
        browserNode = (
            slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(
                csaSeqNode
            )
        )
        proxyTableNode = browserNode.GetProxyNode(csaSeqNode)
        # Determine the envelope of CSA values
        allCSAs = self.convertTableNodeToNumpy(mergedTableNode)
        rasCoords = self.getRASFromTable(mergedTableNode)
        lowestCSA = np.min(allCSAs, axis=1)
        highestCSA = np.max(allCSAs, axis=1)
        # Make these back into table a table node so that it can be referenced
        # by plotSeries nodes
        vLowestCol = vtk.vtkFloatArray()
        vLowestCol.SetName("MinCSA")
        vHighestCol = vtk.vtkFloatArray()
        vHighestCol.SetName("MaxCSA")
        vSupCoordCol = vtk.vtkFloatArray()
        vSupCoordCol.SetName("S_Coordinate")
        for idx in range(lowestCSA.size):
            vLowestCol.InsertNextValue(lowestCSA[idx])
            vHighestCol.InsertNextValue(highestCSA[idx])
            vSupCoordCol.InsertNextValue(rasCoords[idx, 2])
        # Create table node to hold envelope
        envTableNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTableNode", slicer.mrmlScene.GenerateUniqueName("CSAEnvelopeTable")
        )
        envTableNode.AddColumn(vLowestCol)
        envTableNode.AddColumn(vHighestCol)
        envTableNode.AddColumn(vSupCoordCol)
        # Make plot series for each of these
        lowPlotSeries = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "MinCSA"
        )
        lowPlotSeries.SetAndObserveTableNodeID(envTableNode.GetID())
        lowPlotSeries.SetXColumnName("S_Coordinate")
        lowPlotSeries.SetYColumnName("MinCSA")
        lowPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        gray = [0.75, 0.75, 0.75]
        lowPlotSeries.SetColor(gray)
        highPlotSeries = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "MaxCSA"
        )
        highPlotSeries.SetAndObserveTableNodeID(envTableNode.GetID())
        highPlotSeries.SetXColumnName("S_Coordinate")
        highPlotSeries.SetYColumnName("MaxCSA")
        highPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        highPlotSeries.SetColor(gray)
        # Make plot series for dynamic one using the proxy node
        dynPlotSeries = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "FrameCSA"
        )
        dynPlotSeries.SetAndObserveTableNodeID(proxyTableNode.GetID())
        dynPlotSeries.SetXColumnName("S_coord")
        dynPlotSeries.SetYColumnName("Cross-section area")
        dynPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        dynPlotSeries.SetColor([0, 0, 1])
        # Set up plotChart
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        plotChartNode.SetName(slicer.mrmlScene.GenerateUniqueName("CSA_Dynamic"))
        for plotSeries in [lowPlotSeries, highPlotSeries, dynPlotSeries]:
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeries.GetID())
        plotChartNode.SetXAxisTitle("S-Coordinate (Inferior --> Superior, mm)")
        plotChartNode.SetYAxisTitle("Cross-sectional Area mm^2")
        return plotChartNode, (lowPlotSeries, highPlotSeries, dynPlotSeries)

    def convertTableNodeToNumpy(self, tableNode, skipRASCols=True):
        """Hacky conversion, beware of errors or weird effects if you
        input a table with anything other than straight numbers in equal
        length columns...
        Added skipMultiComponentCols as an easy way to ignore RAS coord col.
        """

        temp_npcols = []
        for idx in range(tableNode.GetTable().GetNumberOfColumns()):
            vtkArr = tableNode.GetTable().GetColumn(idx)
            if skipRASCols and vtkArr.GetNumberOfComponents() > 1:
                continue
            elif skipRASCols and vtkArr.GetName() in ["R_coord", "A_coord", "S_coord"]:
                continue
            npArr = numpy_support.vtk_to_numpy(vtkArr)
            temp_npcols.append(npArr)
        # Stack them all into the final numpy array
        arr = np.column_stack(temp_npcols)
        return arr

    def getRASFromTable(self, tableNode):
        """Most of the tables this module generates have a final RAS coordinate
        column or columns for the centerline locations, this picks that off and returns
        the coordinates as a Nx3 numpy array
        """
        nCols = tableNode.GetTable().GetNumberOfColumns()
        lastCol = tableNode.GetTable().GetColumn(nCols - 1)
        if lastCol.GetNumberOfComponents() == 3:
            rasArr = numpy_support.vtk_to_numpy(lastCol)
        elif lastCol.GetNumberOfComponents() == 1 and lastCol.GetName() == "S_coord":
            vTable = tableNode.GetTable()
            r = numpy_support.vtk_to_numpy(vTable.GetColumnByName("R_coord"))
            a = numpy_support.vtk_to_numpy(vTable.GetColumnByName("A_coord"))
            s = numpy_support.vtk_to_numpy(vTable.GetColumnByName("S_coord"))
            rasArr = np.column_stack((r, a, s))
        return rasArr

    def saveVolumeOutputTableNodeToFile(self, tableNode, saveDir):
        """ """
        csvFileName = pathlib.Path("LimitedAirwayVolumes.csv")
        csvFilePath = pathlib.Path.joinpath(saveDir, csvFileName)
        slicer.util.saveNode(tableNode, str(csvFilePath))

    def consolidateCSATables(self, tableSequence):
        """Gather the cross-sectional area table column across
        each frame and merge it into a unified table.
        Also, since we now want to use the S coordinate for
        plotting, and since the X value for plots needs to be
        in its own column of the table, we will separate the
        R, A, and S coordinates into their own columns rather
        than keeping them merged. (NOTE: this also means we
        need to update the max/min of CSA code to make sure
        to ignore the coord columns)
        """
        mergedTableNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTableNode", "MergedCSATable"
        )
        for frameIdx in range(tableSequence.GetNumberOfDataNodes()):
            tableNode = tableSequence.GetNthDataNode(frameIdx)
            column = tableNode.GetTable().GetColumnByName("Cross-section area")
            newCol = vtk.vtkFloatArray()
            newCol.SetName(f"CSA (#{frameIdx})")
            for dataIdx in range(column.GetNumberOfValues()):
                newCol.InsertNextValue(column.GetValue(dataIdx))
            # Add it to the table
            mergedTableNode.AddColumn(newCol)
        # Also add the RAS coordinates of from the centerline, associated with each row
        vTable = tableNode.GetTable()
        rCol = vTable.GetColumnByName("R_coord")
        aCol = vTable.GetColumnByName("A_coord")
        sCol = vTable.GetColumnByName("S_coord")
        # Add RAS cols to the table
        mergedTableNode.AddColumn(rCol)
        mergedTableNode.AddColumn(aCol)
        mergedTableNode.AddColumn(sCol)
        return mergedTableNode

    def saveMergedCsaTableNodeToFile(self, tableNode, saveDir):
        """ """
        csvFileName = pathlib.Path("MergedAirwayCSAs.csv")
        csvFilePath = pathlib.Path.joinpath(saveDir, csvFileName)
        slicer.util.saveNode(tableNode, str(csvFilePath))

    def addVolumesToTable(self, limitedSegmentVolumes, tableNode):
        """Add the segment volume data to the given table node as a
        new column
        """
        column = vtk.vtkFloatArray()
        column.SetName("LimitedSegmentVolume_cm3")
        for vol in limitedSegmentVolumes:
            column.InsertNextValue(vol)
        tableNode.AddColumn(column)

    def getLimitedSegmentVolume(
        self, segmentName, segmentationNode, limitingSegmentID, limitingSegmentationNode
    ):
        """Limit a copy of the segment in segmentationNode with name segmentName by
        intersection with the segment from limitingSegmentationNode (could be same or
        different) with segmentID limitingSegmentID, and return the limited segment
        volume in cm3.
        """
        limitedSegmentName = f"{segmentName}_limited"
        dupSegID = self.duplicateSegment(
            segmentName, segmentationNode, limitedSegmentName
        )
        self.limitSegmentRegion(
            dupSegID, segmentationNode, limitingSegmentID, limitingSegmentationNode
        )
        volumeMeasurementNameCm3 = "LabelmapSegmentStatisticsPlugin.volume_cm3"
        segStatsLogic = self.generateSegmentVolumeStats(segmentationNode)
        segStats = segStatsLogic.getStatistics()
        volCm3 = segStats[limitedSegmentName, volumeMeasurementNameCm3]

        return volCm3

    def duplicateSegment(self, segmentNameToDup, segmentationNode, dupName: str):
        """ """
        segID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(
            segmentNameToDup
        )
        dupSegID = copy_segment(dupName, segmentationNode, segmentIDToCopy=segID)
        return dupSegID

    def limitSegmentRegion(
        self,
        segmentIDToLimit,
        segmentationNode,
        segmentIDToIntersectWith,
        otherSegmentationNode=None,
    ):
        """Limit a given segment in a given segmentation to only that area which intersects with the another segment, possibly from
        another segmentation
        """
        if otherSegmentationNode is None:
            otherSegmentationNode = segmentationNode
        # Check whether the segment to intersect with is in a different segmentation or the same segmentation
        if otherSegmentationNode != segmentationNode:
            # Segment to intersect with needs to be copied in from the other segmentation node
            segmentationNode.GetSegmentation().CopySegmentFromSegmentation(
                otherSegmentationNode.GetSegmentation(), segmentIDToIntersectWith
            )
        # Do intersection
        intersect_segment(
            segmentIDToLimit, segmentIDToIntersectWith, segmentationNode
        )  # use helper function
        if otherSegmentationNode != segmentationNode:
            # Remove copied segment
            segmentationNode.RemoveSegment(segmentIDToIntersectWith)

    def generateSegmentVolumeStats(self, segmentationNode):
        """Compute segment volumes"""
        import SegmentStatistics

        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        spn = segStatLogic.getParameterNode()
        spn.SetParameter("Segmentation", segmentationNode.GetID())
        spn.SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", "False")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.enabled", "True")
        spn.SetParameter("ClosedSurfaceSegmentStatisticsPlugin.enabled", "False")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.voxel_count.enabled", "True")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.volume_mm3.enabled", "False")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.volume_cm3.enabled", "True")
        # Actually do the computation
        segStatLogic.computeStatistics()
        return segStatLogic

    def setCompatibleSeqIndexing(self, newSeqNode, seqNodeToMatch):
        """Sets indexing to be compatible by matching the index
        name, type, and units to match an existing sequence node.
        This is necessary for them to share a single sequence browser
        and be synchronized.
        """
        newSeqNode.SetIndexName(seqNodeToMatch.GetIndexName())
        newSeqNode.SetIndexType(seqNodeToMatch.GetIndexType())
        newSeqNode.SetIndexUnit(seqNodeToMatch.GetIndexUnit())

    def getSegmentInfoFromSequence(
        self, segmentationSequence: vtkMRMLSequenceNode
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        """This is the first function I've ever had ChatGPT write and actually
        used (with minimal modification after testing). It loops over the
        elements of the input segmentation sequence node and examines which
        segments are present in each frame. It returns a list of segments
        which are present in every frame, and, separately, a dictionary
        of segments which are present in only some frames, where the key
        is the segment name, and the value is a list of the frames where
        that segment name appears.
        This function does NOT check that the segment ID is consistent, and
        indeed, it may vary across frames, so segment IDs should always be
        retrieved from the segment name in each frame.
        """
        # Initialize dictionaries to store segment presence across frames
        segmentPresence = {}
        totalFrames = segmentationSequence.GetNumberOfDataNodes()
        # Iterate through each item in the sequence
        for i in range(totalFrames):
            # Get the segmentation node from the sequence
            segmentationNode = segmentationSequence.GetNthDataNode(i)
            # Get the segmentation object
            segmentation = segmentationNode.GetSegmentation()
            # Iterate through each segment in the segmentation node
            for j in range(segmentation.GetNumberOfSegments()):
                segmentID = segmentation.GetNthSegmentID(j)
                segment = segmentation.GetSegment(segmentID)
                segmentName = segment.GetName()
                # Initialize segment presence if not already done
                if segmentName not in segmentPresence:
                    segmentPresence[segmentName] = []
                # Record the frame index where the segment is present
                segmentPresence[segmentName].append(i)
        # Determine segments present in every frame and those present in only some frames
        segmentsInEveryFrame = []
        segmentsInSomeFramesDict = {}
        for segmentName, frames in segmentPresence.items():
            if len(frames) == totalFrames:
                segmentsInEveryFrame.append(segmentName)
            else:
                segmentsInSomeFramesDict[segmentName] = frames
        return segmentsInEveryFrame, segmentsInSomeFramesDict

    def processSegmentSelectorTextToNameAndFrames(self, optionString):
        """Process the segment name option string back into a segment
        name and a list of frames. If present in all frames, return
        None for frame list.  If divider is chosen or an empty string
        is input, return None,None
        """
        if optionString == SEGMENT_NAME_DIVIDER:
            return None, None
        # Pattern is ^SegmentName (\[(frameList)\])?
        frameListPatt = re.compile(
            r"""\ \(in\ \[ # match " (in ["
                ([0-9, ]*) # match the comma-separated frame num list
                ]\)$ # match "])"
            """,
            re.VERBOSE,
        )
        frMatch = frameListPatt.search(optionString)
        if frMatch:
            # There is a list of frames
            frames = [int(frStr) for frStr in frMatch.group(1).split(",")]
            namePattForPartial = re.compile("(.*)(?! \(in \[)")
            # pattern matches everything before " (in ["
            nameMatch = namePattForPartial.match(optionString)
            segmentName = nameMatch.group(1)
        else:
            # This segment is in all frames
            segmentName = optionString
            frames = None
        return segmentName, frames

    def buildOptionStrings(self, segmentsList, partialSegmentDict):
        """Build the list of option strings for the segment chooser
        combobox.  The segments which are in all frames go on top,
        then a divider, then segments which are only in some frames.
        """
        optionStrings = [segmentName for segmentName in segmentsList]
        optionStrings.append(SEGMENT_NAME_DIVIDER)
        for segmentName, frames in partialSegmentDict.items():
            frameList = ", ".join([f"{frame}" for frame in frames])
            optionStr = f"{segmentName} (in [{frameList}])"
            optionStrings.append(optionStr)
        return optionStrings

    def updateComboBoxOptions(self, comboBox, newOptions: List[str]) -> bool:
        """Update a qComboBox with the list of new options, only if the
        list of options is different from the current list. Returns True
        if the list of options changed, and returns False otherwise.
        """
        currentOptions = [comboBox.itemText(i) for i in range(comboBox.count)]
        if currentOptions != newOptions:
            comboBox.clear()
            comboBox.addItems(newOptions)
            listChanged = True
        else:
            listChanged = False
        return listChanged

    def runCrossSectionalAnalysis(
        self,
        centerline: vtkMRMLMarkupsCurveNode,
        lumenNode: vtkMRMLSegmentationNode,
        lumenSegmentID: str,
        outputTable: vtkMRMLTableNode,
    ):
        """Run the cross-sectional analysis. This is mostly just a wrapper
        for the function in the CrossSectionAnalysis module. However,
        since we now want the S-coordinate available to use for plotting,
        we now split the formerly combined RAS column into 3 individual
        columns named 'R_coord', 'A_coord', and 'S_coord'.
        """
        import CrossSectionAnalysis

        csaLogic = CrossSectionAnalysis.CrossSectionAnalysisLogic()
        csaLogic.inputCenterlineNode = centerline
        csaLogic.lumenSurfaceNode = lumenNode
        csaLogic.currentSegmentID = lumenSegmentID
        csaLogic.outputTableNode = outputTable
        csaLogic.outputPlotSeriesNode = None
        csaLogic.run()
        # csaLogic.updateOutputTable(centerline, outputTable)
        vTable = outputTable.GetTable()
        rasCol = vTable.GetColumnByName("RAS")
        rCol = vtk.vtkFloatArray()
        rCol.SetName("R_coord")
        aCol = vtk.vtkFloatArray()
        aCol.SetName("A_coord")
        sCol = vtk.vtkFloatArray()
        sCol.SetName("S_coord")
        for idx in range(rasCol.GetNumberOfTuples()):
            rCol.InsertNextValue(rasCol.GetComponent(idx, 0))
            aCol.InsertNextValue(rasCol.GetComponent(idx, 1))
            sCol.InsertNextValue(rasCol.GetComponent(idx, 2))
        # Remove combined col and add new ones
        outputTable.GetTable().RemoveColumnByName("RAS")
        outputTable.AddColumn(rCol)
        outputTable.AddColumn(aCol)
        outputTable.AddColumn(sCol)

        return outputTable


#
# Helper functions
#


def copy_segment(newSegmentName, segmentationNode, segmentIDToCopy):
    """Copy an existing segment into a new segment without overwriting"""
    import SegmentEditorEffects

    segmentEditorWidget, segmentEditorNode, segmentationNode = setup_segment_editor(
        segmentationNode
    )
    newSegmentID = segmentationNode.GetSegmentation().AddEmptySegment(newSegmentName)
    segmentEditorNode.SetSelectedSegmentID(newSegmentID)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", SegmentEditorEffects.LOGICAL_COPY)
    effect.setParameter("ModifierSegmentID", segmentIDToCopy)
    # Masking Settings
    effect.setParameter(
        "BypassMasking", 1
    )  # probably not necessary since I adjust masking settings next
    segmentEditorNode.SetMaskMode(segmentationNode.EditAllowedEverywhere)
    segmentEditorNode.MasterVolumeIntensityMaskOff()
    segmentEditorNode.SetOverwriteMode(segmentEditorNode.OverwriteNone)
    # Do the copy
    effect.self().onApply()
    slicer.mrmlScene.RemoveNode(segmentEditorNode)
    return newSegmentID


def intersect_segment(segmentID, segmentIDToIntersectWith, segmentationNode):
    """Keep only the portion of an existing segment which intersects another segment"""
    import SegmentEditorEffects

    segmentEditorWidget, segmentEditorNode, segmentationNode = setup_segment_editor(
        segmentationNode
    )
    segmentEditorNode.SetSelectedSegmentID(segmentID)
    segmentEditorWidget.setActiveEffectByName("Logical operators")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", SegmentEditorEffects.LOGICAL_INTERSECT)
    effect.setParameter("ModifierSegmentID", segmentIDToIntersectWith)
    effect.setParameter(
        "BypassMasking", 1
    )  # will not modify any other segments or apply any intensity masking to the results
    effect.self().onApply()
    slicer.mrmlScene.RemoveNode(segmentEditorNode)


def setup_segment_editor(segmentationNode=None, masterVolumeNode=None):
    """Runs standard setup of segment editor widget and segment editor node"""
    if segmentationNode is None:
        # Create segmentation node
        segmentationNode = slicer.vtkMRMLSegmentationNode()
        slicer.mrmlScene.AddNode(segmentationNode)
        segmentationNode.CreateDefaultDisplayNodes()
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
    slicer.mrmlScene.AddNode(segmentEditorNode)
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(segmentationNode)
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    if masterVolumeNode:
        if slicer.app.majorVersion > 4:
            # Method name change
            segmentEditorWidget.setSourceVolumeNode(masterVolumeNode)
        else:
            # old method name
            segmentEditorWidget.setMasterVolumeNode(masterVolumeNode)
    return segmentEditorWidget, segmentEditorNode, segmentationNode
