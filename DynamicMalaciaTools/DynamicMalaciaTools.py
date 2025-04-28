import logging
import os
from typing import Annotated, Optional, List, Tuple, Dict, Union

import vtk
import qt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
    FloatRange,
    RangeBounds,
)

import json
import numpy as np
import re
import traceback
import pathlib

try:
    import pyvista as pv
except ImportError:
    slicer.util.pip_install("pyvista")
    import pyvista as pv

from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from vtk.util import numpy_support
from collections import namedtuple
from slicer.util import (
    warningDisplay,
    arrayFromVolume,
    arrayFromVolumeModified,
    arrayFromMarkupsControlPoints,
    createProgressDialog,
)
import slicer.util

from slicer import (
    vtkMRMLScalarVolumeNode,
    vtkMRMLSegmentationNode,
    vtkMRMLMarkupsCurveNode,
    vtkMRMLMarkupsFiducialNode,
    vtkMRMLMarkupsROINode,
    vtkMRMLSequenceNode,
    vtkMRMLSequenceBrowserNode,
    vtkMRMLTableNode,
    vtkMRMLModelNode,
    vtkMRMLPlotChartNode,
    vtkMRMLTextNode,
    vtkMRMLPlotSeriesNode,
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
            "CrossSectionAnalysis",
            "ExtractCenterline",
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
        # Register custom layouts on load
        tempLogic = DynamicMalaciaToolsLogic()
        slicer.app.startupCompleted.connect(lambda: tempLogic.createReviewLayout())


#
# DynamicMalaciaToolsParameterNode
#


@parameterNodeWrapper
class DynamicMalaciaToolsParameterNode:
    """
    The parameters needed by module.
    """

    #### Source Data
    origImageSequenceNode: vtkMRMLSequenceNode = None
    # browserNode: vtkMRMLSequenceBrowserNode = None
    temporalStep: float = 0.1
    #### Initial Segmentation Phase
    initialCropBox: vtkMRMLMarkupsROINode = None
    initialMinIP: vtkMRMLScalarVolumeNode = None
    initialMinIPAirwaySegmentation: vtkMRMLSegmentationNode = None
    initialMinIPSegID: str = ""
    initialCroppedImageSeq: vtkMRMLSequenceNode = None
    initialMinIPSegmentationFinishedFlag: bool = False
    initialAirSegSeq: vtkMRMLSequenceNode = None
    carinaLocationsSeqNode: vtkMRMLSequenceNode = None
    #### Registered Segmentation Phase
    carinaAlignmentTransformSeq: vtkMRMLSequenceNode = None
    currentlyAligned: bool = False
    alignedCropBox: vtkMRMLMarkupsROINode = None
    alignedMinIP: vtkMRMLScalarVolumeNode = None
    alignedMinIPSegmentation: vtkMRMLSegmentationNode = None
    alignedMinIPSegID: str = ""
    alignedCroppedImageSeq: vtkMRMLSequenceNode = None
    alignedAirSegSeq: vtkMRMLSequenceNode = None

    #### Centerline Phase
    cline_InputSurfaceNode: Union[vtkMRMLModelNode, vtkMRMLSegmentationNode, None] = (
        None
    )
    cline_InputSegmentID: str = ""
    cline_Endpoints: vtkMRMLMarkupsFiducialNode
    centerlineStepSizeMm: float = 1.0
    # outputs:
    rawCenterline: vtkMRMLMarkupsCurveNode = None
    smoothCenterline: vtkMRMLMarkupsCurveNode = None

    #### Quantification Phase
    q_airwaySegmentationSequence: vtkMRMLSequenceNode
    q_smoothedCenterline: vtkMRMLMarkupsCurveNode
    q_segmentIdx: int = 0
    # ^^ the index of the segment to quantify in q_airwaySegmentationSequence segmentations
    q_carinaDistanceList: List[float]
    q_frameTimes: List[float]

    volumeQuantificationBox: vtkMRMLMarkupsROINode
    # outputs:
    csaTable: vtkMRMLTableNode
    arTable: vtkMRMLTableNode
    longAxisTable: vtkMRMLTableNode
    shortAxisTable: vtkMRMLTableNode
    volTable: vtkMRMLTableNode
    slicingInfoTable: vtkMRMLTableNode  # points, plane normals
    quantDataTextNode: vtkMRMLTextNode  # json of the raw output from runQuantification
    # table sequences used in plotting
    csaTableSeq: vtkMRMLSequenceNode
    arTableSeq: vtkMRMLSequenceNode
    longTableSeq: vtkMRMLSequenceNode
    shortTableSeq: vtkMRMLSequenceNode
    # Envelope data tables
    csaEnvTable: vtkMRMLTableNode
    arEnvTable: vtkMRMLTableNode
    longEnvTable: vtkMRMLTableNode
    shortEnvTable: vtkMRMLTableNode
    #
    sliceIndicatorTable: vtkMRMLTableNode

    #### Review Phase
    sliceSliderIdx: Annotated[float, WithinRange(0, 100)] = 0
    sequenceSliderIdx: Annotated[float, WithinRange(0, 40)] = 0
    # ^^ these indexes need to be coerced to int wherever used!
    carinaDistPlotRangeMm: Annotated[FloatRange, RangeBounds(0.0, 100.0)] = FloatRange(
        5.0, 60.0
    )
    updatingCarinaPlotRange: bool = False
    maxY_CSA: float = 130.0
    # plot handles?? what else here?
    crossSectionModel: vtkMRMLModelNode
    longAxisLine: vtkMRMLMarkupsCurveNode
    shortAxisLine: vtkMRMLMarkupsCurveNode

    csaPlotChartNode: vtkMRMLPlotChartNode
    csaPlotSeriesMaxCSA: vtkMRMLPlotSeriesNode
    csaPlotSeriesMinCSA: vtkMRMLPlotSeriesNode
    csaPlotSeriesCurSlice: vtkMRMLPlotSeriesNode
    csaPlotSeriesDynCSA: vtkMRMLPlotSeriesNode
    arPlotChartNode: vtkMRMLPlotChartNode
    arPlotSeriesMaxAR: vtkMRMLPlotSeriesNode
    arPlotSeriesMinAR: vtkMRMLPlotSeriesNode
    arPlotSeriesCurSlice: vtkMRMLPlotSeriesNode
    arPlotSeriesDynAR: vtkMRMLPlotSeriesNode
    arAxesPlotChartNode: vtkMRMLPlotChartNode
    arAxesPlotSeriesLongMax: vtkMRMLPlotSeriesNode
    arAxesPlotSeriesLongMin: vtkMRMLPlotSeriesNode
    arAxesPlotSeriesShortMax: vtkMRMLPlotSeriesNode
    arAxesPlotSeriesShortMin: vtkMRMLPlotSeriesNode
    arAxesPlotSeriesCurSlice: vtkMRMLPlotSeriesNode
    arAxesPlotSeriesShortDyn: vtkMRMLPlotSeriesNode
    arAxesPlotSeriesLongDyn: vtkMRMLPlotSeriesNode
    volPlotChartNode: vtkMRMLPlotChartNode

    # display checkboxes
    showEnvCurvesFlag: bool = True
    showCurrentCurveFlag: bool = True
    showCurrentSlice3DFlag: bool = True
    showARLinesFlag: bool = True
    showSliceIndicatorOnPlotsFlag: bool = True
    showLegendFlag: bool = True

    # Save location
    saveDirectory: pathlib.Path

    ### Some flags
    updatingSliceIdx = False


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
        self.quantData = None
        self.browserObservation = None
        self.updatingOrigSeq = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/DynamicMalaciaTools.ui"))
        self.layout.addWidget(uiWidget)
        ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui = ui

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
        ui.origImageSequenceNodeSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLScalarVolumeNode"
        )
        ui.carinaAlignmentTransformSeqSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLLinearTransformNode"
        )
        ui.carinaLocationsSeqSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLMarkupsFiducialNode"
        )
        ui.airwaySegmentationSequenceSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLSegmentationNode"
        )
        ui.croppedImageSequenceSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLScalarVolumeNode"
        )
        ui.q_airwaySegmentationsSelector.addAttribute(
            "vtkMRMLSequenceNode", "DataNodeClassName", "vtkMRMLSegmentationNode"
        )

        #### Connections
        # qMRMLSegmentSelectorWidgets cannot yet be handled via the parameter node
        # wrapper, so must still be handled manually here
        ui.centerlineInputSurfaceSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.onCenterlineInputSurfaceSelectionChange,
        )
        ui.centerlineSegmentSelectorWidget.connect(
            "currentSegmentChanged(QString)",
            self.onCenterlineSegmentSelectionChange,
        )
        # Anything else which needs to be handled via a separate connection
        # self.ui.origImageSequenceNodeSelector.connect(
        #    "currentNodeChanged(vtkMRMLNode*)",
        #    self.onOrigTest,  # self.onOrigImageSequenceSelectionChange
        # )

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )
        #  we should also update the parameter node when a scene is loaded
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.StartImportEvent,
            self.onStartSceneLoading,
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndImportEvent, self.onSceneLoaded
        )

        # Buttons
        ui.createNewVolumeQuantROIButton.clicked.connect(
            self.onCreateNewVolumeQuantROIButtonClick
        )
        ui.createNewInitialCropROIButton.clicked.connect(
            self.onCreateNewInitialCropROIButtonClick
        )
        ui.createNewAlignedROIButton.clicked.connect(
            self.onCreateNewAlignedCropROIButtonClick
        )
        ui.generateInitialSegmentationsButton.clicked.connect(
            self.onGenerateInitialSegmentationsButtonClick
        )
        ui.findCarinaLocationsButton.clicked.connect(
            self.onFindCarinaLocationsButtonClick
        )
        ui.alignCarinasButton.clicked.connect(self.onAlignCarinasButtonClick)
        ui.createAlignedMinIPButton.clicked.connect(
            self.onRecropImagesAndCreateAlignedMinIPButtonClick
        )
        ui.transferSegmentationButton.clicked.connect(
            self.onTransferSegmentationButtonClick
        )
        ui.transferToAllFramesButton.clicked.connect(
            self.onTransferToAllFramesButtonClick
        )

        ui.findAndSmoothCenterlineButton.clicked.connect(
            self.onFindAndSmoothCenterlineButtonClick
        )
        ui.applyXRangeButton.clicked.connect(self.onApplyXRangeButtonClick)
        ui.applyYMaxButton.clicked.connect(self.onApplyYMaxCSAButtonClick)
        ui.switchToReviewLayoutButton.clicked.connect(self.logic.switchToReviewLayout)
        ui.switchToRegularLayoutButton.clicked.connect(self.logic.switchToRegularLayout)

        ui.saveResultsButton.clicked.connect(self.onSaveResultsButtonClick)
        ui.cropImagesAndCreateMinIPButton.clicked.connect(
            self.onCropImagesAndCreateMinIPButtonClick
        )
        ui.runQuantificationButton.clicked.connect(self.onRunQuantificationButtonClick)

        # Connect slice slider
        ui.sliceSlider.connect("valueChanged(double)", self.onSliceSliderValueChange)

        # ui.carinaDistPlotRangeSlider.connect(
        #    "valuesChanged(double,double)", self.onCarinaDistPlotRangeSliderChange
        # )
        # Hide unused slider and label
        ui.timeStepSlider.hide()
        ui.timeIndexLabel.hide()

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Ensure that the custom review layout is created before it is needed
        self.logic.createReviewLayout()

        # Plot series didn't used to be stored in the parameter node. For backwards
        # compatibility, connect these if the plots exist but the series are not in
        # the parameter node
        pn = self._parameterNode
        if pn.csaPlotChartNode and not pn.csaPlotSeriesDynCSA:
            # Plot series parameter node references need to be updated
            pn.arPlotSeriesMaxAR = pn.arPlotChartNode.GetNthPlotSeriesNode(0)
            pn.arPlotSeriesMinAR = pn.arPlotChartNode.GetNthPlotSeriesNode(1)
            pn.arPlotSeriesCurSlice = pn.arPlotChartNode.GetNthPlotSeriesNode(2)
            pn.arPlotSeriesDynAR = pn.arPlotChartNode.GetNthPlotSeriesNode(3)

            pn.arAxesPlotSeriesLongMax = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(0)
            pn.arAxesPlotSeriesLongMin = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(1)
            pn.arAxesPlotSeriesShortMax = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(2)
            pn.arAxesPlotSeriesShortMin = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(3)
            pn.arAxesPlotSeriesCurSlice = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(4)
            pn.arAxesPlotSeriesShortDyn = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(5)
            pn.arAxesPlotSeriesLongDyn = pn.arAxesPlotChartNode.GetNthPlotSeriesNode(6)
            # longDyn and shortDyn could possibly be switched here, but I think this is
            # correct for backwards compatibility. New cases will not pass through this step

            pn.csaPlotSeriesMaxCSA = pn.csaPlotChartNode.GetNthPlotSeriesNode(0)
            pn.csaPlotSeriesMinCSA = pn.csaPlotChartNode.GetNthPlotSeriesNode(1)
            pn.csaPlotSeriesCurSlice = pn.csaPlotChartNode.GetNthPlotSeriesNode(2)
            pn.csaPlotSeriesDynCSA = pn.csaPlotChartNode.GetNthPlotSeriesNode(3)
        # Make sure segment list is initialized if there is an
        # initial segmentation sequence
        # self.onSegmentationSequenceSelectorChange(
        #    self._parameterNode.segmentationSequence
        # )

    def updateBrowserObservation(self):
        """Make sure browser is observed and linked to the current browser.
        Should be called when runQuantification is run. Removes any
        existing browser observation (in case the original sequence
        has changed).
        """
        if self.browserObservation:
            self.removeBrowserObservation()
        self.addBrowserObservation(self.getBrowserNode())

    def addBrowserObservation(self, browser):
        """Register onFrameChange callback to update cross section and arAxes
        when browser frame changes
        """
        # pn = self._parameterNode
        if not browser:
            logging.warning("No browser to observe!")
            return
        if self.browserObservation:
            raise RuntimeError(
                "Browser is already observed, blocking you from adding another layer of observation!!"
            )
            # Consider: Could also potentially auto-remove here before adding new one...
        self.addObserver(
            browser,
            browser.ProxyNodeModifiedEvent,
            self.onFrameChange,
        )
        # Store record of observation so it can be removed as needed
        logging.info("Adding browser observation.")
        self.browserObservation = (
            browser,
            browser.ProxyNodeModifiedEvent,
            self.onFrameChange,
        )

    def removeBrowserObservation(self):
        """Unregister observation of browser changes using stored info
        about the observation.
        """
        if not self.browserObservation:
            logging.warning("No stored browser observation to remove!")
            return
        self.removeObserver(*(self.browserObservation))
        self.browserObservation = None
        logging.info("Removed browser observation.")

    def getBrowserNode(self):
        """Get the the browser node linked to the original sequence node. If
        a different browser node is needed, code can be added here to allow
        a way of selecting it.
        """
        pn = self._parameterNode
        if not pn.origImageSequenceNode:
            return None
        browser = getFirstBrowser(pn.origImageSequenceNode)
        return browser

    def getProxyNode(self, seqNode):
        """Get the proxy node for the sequence node from the browser"""
        browser = self.getBrowserNode()
        if browser is None:
            return None
        return browser.GetProxyNode(seqNode)

    def onCreateNewVolumeQuantROIButtonClick(self):
        """Create a new volume quantification ROI node.  Base it on
        the initial Crop ROI node.
        """
        pn = self._parameterNode
        pn.volumeQuantificationBox = self.logic.initializeVolumeQuantBox(
            pn.alignedCropBox
        )

    def onCreateNewInitialCropROIButtonClick(self):
        """Create a new ROI for cropping the original image sequence (facilitates minIP
        segmentation, speeds computation)
        """
        pn = self._parameterNode
        # Create or update the initial crop ROI
        origProxyNode = self.getProxyNode(pn.origImageSequenceNode)
        pn.initialCropBox = self.logic.initializeInitCropBox(
            refVolNode=origProxyNode, initialCropBox=None
        )

    def onCreateNewAlignedCropROIButtonClick(self):
        """Create a new ROI for cropping the aligned original image sequence"""
        # copy the initial crop box, change color, then hide initial ROI (to avoid confusion?)
        pn = self._parameterNode
        pn.alignedCropBox = self.logic.initializeAlignedCropBox(pn.initialCropBox)

    def onGenerateInitialSegmentationsButtonClick(self):
        """Using initial minIP segmentation and cropped image sequence,
        generate the initial airway segmentation sequence by intersection
        of minIP airway with <-250 HU region in each frame or sequence.
        """
        pn = self._parameterNode
        if not pn.initialMinIPSegmentationFinishedFlag:
            warningDisplay(
                "Initial minIP segmentation is not marked finished!  Check box and press button again if ready!"
            )
            return
        pn.initialAirSegSeq = self.logic.generateInitialSegmentationSequence(
            pn.initialCroppedImageSeq,
            pn.initialMinIPAirwaySegmentation,
            pn.initialMinIPSegID,
        )
        # Hide the minIP segmentation (to be able to see initial frame seg better)
        self.logic.setItemsVisibility([pn.initialMinIPAirwaySegmentation], False)

    def onFindCarinaLocationsButtonClick(self):
        """Use the initial airway segmentation sequence to find the sequence
        of carina locations.
        """
        pn = self._parameterNode
        pn.carinaLocationsSeqNode = self.logic.findCarinas(
            pn.initialAirSegSeq, outputCarinaSeq=pn.carinaLocationsSeqNode
        )
        # Lock carina points against accidental movement
        self.logic.setCarinaLocationsLockedStatus(
            pn.carinaLocationsSeqNode, lockedFlag=True
        )

    def onAlignCarinasButtonClick(self):
        """Use carina locations to create a sequence of linear translation-only
        transforms and soft-apply them to both the original and cropped image
        sequences
        Modified to function as a toggle to go back and forth between aligned
        and un-aligned states.  This should make any manual adjustment to carina
        locations more straightforward: Align to test alignment, then unalign to
        adjust, and align to check improvement.
        """
        pn = self._parameterNode
        seqList = [
            pn.origImageSequenceNode,
            pn.initialCroppedImageSeq,
            pn.initialAirSegSeq,
            pn.carinaLocationsSeqNode,
        ]
        if pn.currentlyAligned:
            # Toggle to un-aligned state by removing transforms
            self.logic.unapplyTransformsOnProxies(seqList)
            # Toggle string on button
            self.ui.alignCarinasButton.text = "Align Carinas"
            self.ui.alignCarinasButton.toolTip = "Update carina transforms from landmark locations, and align carina locations on original images, cropped images, segmentations, and landmark points."
            # Toggle currentlyAligned flag
            pn.currentlyAligned = not pn.currentlyAligned
        else:
            # Toggle to aligned state
            # First, update carina transforms
            pn.carinaAlignmentTransformSeq = self.logic.getCarinaTransforms(
                pn.carinaLocationsSeqNode, pn.carinaAlignmentTransformSeq
            )
            # Soft apply alignment to original and cropped images (and segmentation?)
            self.logic.applyTransformToProxies(seqList, pn.carinaAlignmentTransformSeq)
            # Toggle string on button
            self.ui.alignCarinasButton.text = "Undo Align Carinas"
            self.ui.alignCarinasButton.toolTip = "Remove carina transforms from original images, cropped images, segmentations, and landmark points."
            # Toggle currentlyAligned flag
            pn.currentlyAligned = not pn.currentlyAligned

    def onTransferSegmentationButtonClick(self):
        """Use one of the initial segmentations to seed the aligned minIP segmentation.
        Grow it by a margin to capture variations.  Then it's up to the user to fine tune.
        NOTE: I don't want to just merge all initial airway segmentations because that would end
        up including any bad regions, thereby negating one of the main points of having a
        second round of segmentations after carina alignment.
        """
        pn = self._parameterNode
        # Ensure that segmentations are aligned before transferring!
        if not pn.currentlyAligned:
            self.onAlignCarinasButtonClick()
            logging.warning(
                "Aligning images before transfering segmentation for aligned MinIP seg!"
            )
        #
        pn.alignedMinIPSegmentation, pn.alignedMinIPSegID = (
            self.logic.transferSegmentationToAlignedMinIP(
                pn.initialAirSegSeq, pn.alignedMinIP, pn.alignedMinIPSegmentation
            )
        )
        # Hide the transferred
        self.logic.setItemsVisibility(
            [
                pn.initialAirSegSeq,
            ],
            showFlag=False,
        )
        # Also set it as the default input surface for centerline determination
        pn.cline_InputSurfaceNode = pn.alignedMinIPSegmentation
        pn.cline_InputSegmentID = pn.alignedMinIPSegID

    def onTransferToAllFramesButtonClick(self):
        """Using the aligned minIP segmentation, generate the aligned airway segmentation
        for all frames. Threshold each frame and intersect with the aligned minIP airway
        segment.
        """
        pn = self._parameterNode
        pn.alignedMinIPSegmentation
        pn.alignedMinIPSegID
        pn.alignedCroppedImageSeq
        pn.alignedAirSegSeq = self.logic.generateAlignedSegmentationSequence(
            pn.alignedCroppedImageSeq, pn.alignedMinIPSegmentation, pn.alignedMinIPSegID
        )
        # Also make this the default airway sequence for quantification
        pn.q_airwaySegmentationSequence = pn.alignedAirSegSeq
        # Hide the initial segmentation and both minip segmentations
        self.logic.setItemsVisibility(
            [
                pn.initialAirSegSeq,
                pn.alignedMinIPSegmentation,
                pn.initialMinIPAirwaySegmentation,
            ],
            showFlag=False,
        )

    def onCropImagesAndCreateMinIPButtonClick(self):
        """Crop image sequence using initialCropROI on the original images (no transforms),
        and create the initialMinIP image.  Go ahead and create the initialMinIPSegmentation
        as well, and create the initMinIPAirway segment by thresholding.
        """
        pn = self._parameterNode
        # Check for needed inputs
        if not pn.origImageSequenceNode:
            warningDisplay(
                "Missing original image series input! Cropping and MinIP creation canceled!"
            )
            return
        if not pn.initialCropBox:
            warningDisplay(
                "No initial crop box specified! Cropping and MinIP creation canceled!"
            )
            return
        # Turn off interactive adjustment while cropping (and after)
        pn.initialCropBox.GetDisplayNode().SetHandlesInteractive(False)
        # Inputs seem OK
        outputs = self.logic.runCropImagesAndCreateMinIP(
            pn.origImageSequenceNode,
            pn.initialCropBox,
            transformSeq=None,
        )
        # Store outputs
        pn.initialCroppedImageSeq = outputs["croppedSeq"]
        pn.initialMinIP = outputs["minIP"]
        # No need to see the crop box anymore
        pn.initialCropBox.GetDisplayNode().SetVisibility(False)
        # Initialize the initialMinIPSegmentation also
        pn.initialMinIPAirwaySegmentation, segID = self.logic.initializeInitMinIPSeg(
            pn.initialMinIP, pn.initialMinIPAirwaySegmentation
        )
        pn.initialMinIPSegID = segID
        # Center the 3D view (so segment is visible in port)
        self.logic.center3DView()

    def onRecropImagesAndCreateAlignedMinIPButtonClick(self):
        """Using the original images, aligned crop box ROI, and the carina transforms,
        align original images, crop them to ROI, assemble into a sequence, and generate
        a new MinIP image (alignedMinIP).
        """
        pn = self._parameterNode
        # Check for needed inputs
        if not pn.origImageSequenceNode:
            warningDisplay(
                "Missing original image series input! Cropping and MinIP creation canceled!"
            )
            return
        # Reuse initial crop box? Or do we need a different one?
        if not pn.initialCropBox:
            warningDisplay(
                "No initial crop box specified! Cropping and MinIP creation canceled!"
            )
            return
        if not pn.carinaAlignmentTransformSeq:
            warningDisplay(
                "Carina alignment transform sequence missing! Cropping and MinIP creation canceled!"
            )
            return
        # Inputs seem OK
        # No longer need to interact with the crop box
        pn.alignedCropBox.GetDisplayNode().SetHandlesInteractive(False)
        # Do the cropping and create aligned minIP
        outputs = self.logic.runCropImagesAndCreateMinIP(
            origImageSequenceNode=pn.origImageSequenceNode,
            cropBox=pn.alignedCropBox,
            transformSeq=pn.carinaAlignmentTransformSeq,
        )
        pn.alignedCroppedImageSeq = outputs["croppedSeq"]
        pn.alignedMinIP = outputs["minIP"]
        # Hide intial cropbox, initital minIP seg, and initial segmentation
        hideList = [
            pn.initialCropBox,
            pn.initialMinIPAirwaySegmentation,
            pn.alignedCropBox,
        ]
        self.logic.setItemsVisibility(hideList, False)
        # don't hide pn.initialAirSegSeq because it is helpful to see and choose
        # which frame gets transferred in the Transfer step...

    def onFindAndSmoothCenterlineButtonClick(self):
        """Use ExtractCenterline module to find centerline, and then smooth it"""
        ## Gather inputs
        pn = self._parameterNode
        inputSurface = pn.cline_InputSurfaceNode
        inputSegmentID = pn.cline_InputSegmentID
        endPoints = pn.cline_Endpoints
        rawCenterline = pn.rawCenterline
        ## Get centerline
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        try:
            rawCenterline = self.logic.runExtractCenterline(
                inputSurface, inputSegmentID, endPoints, rawCenterline=rawCenterline
            )
        except Exception as e:
            slicer.util.errorDisplay("Failed to compute results: " + str(e))
            traceback.print_exc()
            return
        finally:
            qt.QApplication.restoreOverrideCursor()
        # Lock endpoints so they aren't accidentally moved
        self.logic.setControlPointsLockedStatus(endPoints, lockedFlag=True)
        ## Fix orientation if endpoints were reversed
        carinaMarkup = self.getProxyNode(pn.carinaLocationsSeqNode)
        self.logic.standardizeCenterlineOrientation(rawCenterline, carinaMarkup)
        pn.rawCenterline = rawCenterline
        ## Smooth centerline
        pn.smoothCenterline = self.logic.smoothCenterline(
            pn.rawCenterline, pn.smoothCenterline
        )
        ## Resample to desired step length
        self.logic.resampleCurveNode(
            pn.smoothCenterline,
            pathStepSpacingMm=pn.centerlineStepSizeMm,
            resampledOutputCurveNode=pn.smoothCenterline,
        )
        # Adjust visibility
        self.logic.setItemsVisibility([rawCenterline, endPoints], showFlag=False)
        self.logic.setItemsVisibility([pn.smoothCenterline], showFlag=True)
        self.logic.setControlPointsLockedStatus(pn.smoothCenterline, lockedFlag=True)
        ## Finished
        slicer.util.showStatusMessage(
            "Centerline extraction, smoothing, and resampling is complete.", 3000
        )
        # Also set the default quantification curve to the final smooth curve
        pn.q_smoothedCenterline = pn.smoothCenterline

    def onSaveResultsButtonClick(self):
        """Save quantitative results tables to files. Does not save
        the full scene to avoid duplicating the often very large image
        file sequences. Existing results in this location are overwritten
        without warning. Also captures a screenshot of the current view.

        If the user wants to preserve everything, saving
        the scene as .mrb before exiting Slicer is the preferred mechanism.
        """
        pn = self._parameterNode
        saveDir = pn.saveDirectory
        ### Get screen capture
        import ScreenCapture

        cap = ScreenCapture.ScreenCaptureLogic()
        capturePathName = pathlib.Path(saveDir, "ScreenCapture.png")
        cap.captureImageFromView(None, capturePathName)
        ### Save table files
        fileNameTuples = (
            (pn.csaTable, "CSA_table.csv"),
            (pn.arTable, "AR_table.csv"),
            (pn.slicingInfoTable, "SlicingInfo_table.csv"),
            (pn.volTable, "AirwayVolumes_table.csv"),
        )
        try:
            for dataNode, filename in fileNameTuples:
                filePath = pathlib.Path(saveDir, filename)
                slicer.util.exportNode(dataNode, filePath)
        except Exception as e:
            slicer.util.warningDisplay(
                "Warning, error encountered while saving! Check error log for more details."
            )
            raise e
        logging.info(f'Quantitative results tables sucessfully saved to "{saveDir}"!')
        # Delete the unnecessary and confusing "schema" csv files which are
        # automatically created when exporting or saving table nodes
        for _, filename in fileNameTuples:
            schemaFileName = "".join([filename[:-3], "schema.", filename[-3:]])
            schemaFilePath = pathlib.Path(saveDir, schemaFileName)
            schemaFilePath.unlink()

    def getQuantDataFromJson(self):
        """Recover the quantData from stored json string in parameter node"""
        pn = self._parameterNode
        if not pn.quantDataTextNode:
            return None
        quantData = deserialize_dict_with_ndarrays(pn.quantDataTextNode.GetText())
        return quantData

    def getCurrentCarinaDistance(self):
        """Get the current slice point distance from the carina location."""
        pn = self._parameterNode
        return pn.q_carinaDistanceList[int(pn.sliceSliderIdx)]

    def onRunQuantificationButtonClick(self):
        """Run quantitative analysis on the segmented airway sequence,
        including CSA and AR for every centerline slice, as well as
        volume within a limiting box for each temporal frame."""
        pn = self._parameterNode
        airSegSeq = pn.q_airwaySegmentationSequence
        centerline = pn.q_smoothedCenterline
        volQuantROI = pn.volumeQuantificationBox
        volQuantROI.GetDisplayNode().SetHandlesInteractive(False)
        quantData = self.logic.runQuantification(airSegSeq, centerline, volQuantROI)
        pn.quantDataTextNode = self.logic.createOrUpdateQuantDataTextNode(
            quantData, pn.quantDataTextNode
        )
        # Hide ROI after quantification finishes
        volQuantROI.GetDisplayNode().SetVisibility(False)
        # Also just store quantData in the widget to avoid
        # unnecessary serialization/deserialization cycles
        self.quantData = quantData
        self.makeDataTables(quantData)
        self.makeDataPlots()
        # Change layout and ensure observation will update
        # cross-section stuff when frame changes
        self.initializeReviewSection()

    def makeDataTables(self, quantData=None):
        """After quantification, make all the tables to hold the data
        for plotting/export. If easily available, quantData can be
        passed in as input.  If not input, it is recovered from the
        text node.
        TODO: This takes a LONG time, longer that it seems like it should.
        Investigate.  Might have to do with cycling the browser node
        over and over, which actually, shouldn't be necessary at all,
        right?
        """
        pn = self._parameterNode
        if not quantData:
            quantData = self.quantData
        if not quantData:
            # no data to make tables of
            logging.warning("No data to make tables of! Cancelling data table creation")
            return
        nTables = 15
        progressBar = createProgressDialog(
            value=0,
            maximum=nTables,
            autoClose=True,
            windowTitle="Building Tables",
            labelText="Building tables and table sequences...",
        )
        # Make each data table
        csaData = quantData["csaData"]
        pn.csaTable = tableNodeFromArray(
            csaData, "CSA", transpose=True, outputTableNode=pn.csaTable
        )
        # update progress bar
        progressBar.value = 1
        slicer.app.processEvents()
        pn.csaTable.SetName("CSA")
        arData = quantData["arData"]
        pn.arTable = tableNodeFromArray(
            arData, "AR", transpose=True, outputTableNode=pn.arTable
        )
        pn.arTable.SetName("AR")
        # update progress bar
        progressBar.value = 2
        slicer.app.processEvents()
        # Reshape to a single column (so recognized as 2d by tableNodeFromArray)
        volData = quantData["airwayVolMm3Data"].reshape(-1, 1)
        nFrames = volData.shape[0]
        frameTimes = (pn.temporalStep * np.arange(0, nFrames)).reshape(-1, 1)
        pn.q_frameTimes = frameTimes.ravel().tolist()  # store in widget for easy access
        volTableData = np.hstack((frameTimes, volData))
        pn.volTable = tableNodeFromArray(
            volTableData,
            ["TimeSec", "Volume_mm3"],
            transpose=False,
            outputTableNode=pn.volTable,
        )
        pn.volTable.SetName("AirwayVolumes")
        # update progress bar
        progressBar.value = 3
        slicer.app.processEvents()
        # Assemble slice table data
        slicePointsRAS = quantData["slicePoints"]
        carinaLoc = self.logic.getCarinaLocation(
            pn.carinaLocationsSeqNode,
            pn.currentlyAligned,
            pn.carinaAlignmentTransformSeq,
        )
        carinaDists = self.logic.findCarinaDistances(carinaLoc, slicePointsRAS)
        pn.q_carinaDistanceList = carinaDists.ravel().tolist()  # store in pn
        sliceData = np.hstack(
            [slicePointsRAS, quantData["sliceNormals"], carinaDists.reshape((-1, 1))]
        )
        sliceInfoColNames = ["pR", "pA", "pS", "nR", "nA", "nS", "carinaDistMm"]
        pn.slicingInfoTable = tableNodeFromArray(
            sliceData,
            sliceInfoColNames,
            transpose=False,
            outputTableNode=pn.slicingInfoTable,
        )
        pn.slicingInfoTable.SetName("SlicingInfoTable")
        # update progress bar
        progressBar.value = 4
        slicer.app.processEvents()
        # For plotting, what do we want to see?
        # A dynamic envelope plot of variable vs distance from carina for
        # CSA, AR, Long+Short Axis length (choose which is shown) (plus
        # current slice indicator)
        # For this, we need 4 table sequences (CSA,AR,Long,Short) and the carina distance column
        carinaDistsCol = vtk.vtkFloatArray()
        carinaDistsCol.SetName("DistToCarinaMm")
        for dist in carinaDists:
            carinaDistsCol.InsertNextValue(dist)
        pn.csaTableSeq = tableSeqFromArray(
            np.transpose(csaData),
            xCol=carinaDistsCol,
            browserNode=self.getBrowserNode(),
            dataLabel="CSA",
        )
        pn.csaTableSeq.SetName("CSA_Table_Seq")
        # update progress bar
        progressBar.value = 5
        slicer.app.processEvents()
        # Set the name
        self.getProxyNode(pn.csaTableSeq).SetName("dynCSATable")
        pn.arTableSeq = tableSeqFromArray(
            np.transpose(arData),
            xCol=carinaDistsCol,
            browserNode=self.getBrowserNode(),
            dataLabel="AR",
        )
        pn.arTableSeq.SetName("AR_Table_Seq")
        self.getProxyNode(pn.arTableSeq).SetName("dynARTable")
        # update progress bar
        progressBar.value = 6
        slicer.app.processEvents()
        # Gather just AR axis length data
        longAxisData = quantData["longAxisData"]  # frames,slices,7 (length first)
        longAxisLength = np.squeeze(longAxisData[:, :, 0])
        pn.longAxisTable = tableNodeFromArray(
            longAxisLength,
            "LongAxisLength",
            transpose=True,
            outputTableNode=pn.longAxisTable,
        )
        pn.longAxisTable.SetName("LongAxisLength")
        # update progress bar
        progressBar.value = 7
        slicer.app.processEvents()
        pn.longTableSeq = tableSeqFromArray(
            np.transpose(longAxisLength),
            xCol=carinaDistsCol,
            browserNode=self.getBrowserNode(),
            dataLabel="LongAxis",
        )
        pn.longTableSeq.SetName("LongAxisLength_Seq")
        # update progress bar
        progressBar.value = 8
        slicer.app.processEvents()
        self.getProxyNode(pn.longTableSeq).SetName("dynLongAxisTable")
        shortAxisData = quantData["shortAxisData"]
        shortAxisLength = np.squeeze(shortAxisData[:, :, 0])
        pn.shortAxisTable = tableNodeFromArray(
            shortAxisLength,
            "ShortAxisLength",
            transpose=True,
            outputTableNode=pn.shortAxisTable,
        )
        pn.shortAxisTable.SetName("ShortAxisLength")
        # update progress bar
        progressBar.value = 9
        slicer.app.processEvents()
        pn.shortTableSeq = tableSeqFromArray(
            np.transpose(shortAxisLength),
            xCol=carinaDistsCol,
            browserNode=self.getBrowserNode(),
            dataLabel="ShortAxis",
        )
        pn.shortTableSeq.SetName("ShortAxisLength_Seq")
        self.getProxyNode(pn.shortTableSeq).SetName("dynShortAxisTable")
        # update progress bar
        progressBar.value = 10
        slicer.app.processEvents()
        # Make envelope data tables (CSA, AR, Long Axis Length, Short Axis Length)
        # (these are not sequences, because the envelopes are contstant across frames)
        pn.csaEnvTable = makeEnvTable(
            csaData,
            transposeFlag=True,
            distances=carinaDists,
            dataLabel="CSA",
            outputTableNode=pn.csaEnvTable,
        )
        # update progress bar
        progressBar.value = 11
        slicer.app.processEvents()
        pn.arEnvTable = makeEnvTable(
            arData,
            transposeFlag=True,
            distances=carinaDists,
            dataLabel="AR",
            outputTableNode=pn.arEnvTable,
        )
        # update progress bar
        progressBar.value = 12
        slicer.app.processEvents()
        pn.longEnvTable = makeEnvTable(
            longAxisLength,
            transposeFlag=True,
            distances=carinaDists,
            dataLabel="LongAxisMm",
            outputTableNode=pn.longEnvTable,
        )
        # update progress bar
        progressBar.value = 13
        slicer.app.processEvents()
        pn.shortEnvTable = makeEnvTable(
            shortAxisLength,
            transposeFlag=True,
            distances=carinaDists,
            dataLabel="ShortAxisMm",
            outputTableNode=pn.shortEnvTable,
        )
        # update progress bar
        progressBar.value = 14
        slicer.app.processEvents()
        # I also need
        pn.sliceIndicatorTable = self.createOrUpdateSliceIndicatorTable(
            quantData=quantData, outputTable=pn.sliceIndicatorTable
        )
        # Close progress bar
        progressBar.value = progressBar.maximum

    def makeDataPlots(self):
        """Make all the interactive data plots"""
        # Generate the interactive figures from the tables
        # pn.csaPlotChartNode, (low, high, dyn) = makeEnvPlot()
        pn = self._parameterNode
        browser = self.getBrowserNode()

        nPlots = 4
        progressBar = createProgressDialog(
            value=0,
            maximum=nPlots,
            autoClose=True,
            windowTitle="Building Plots",
            labelText="Building plots and plot series...",
        )
        # CSA
        csaDynTable = browser.GetProxyNode(pn.csaTableSeq)
        (
            pn.csaPlotChartNode,
            pn.csaPlotSeriesMaxCSA,
            pn.csaPlotSeriesMinCSA,
            pn.csaPlotSeriesCurSlice,
            pn.csaPlotSeriesDynCSA,
        ) = self.logic.makeCSAPlot(
            envTable=pn.csaEnvTable,
            dynTable=csaDynTable,
            sliceIndicatorTable=pn.sliceIndicatorTable,
            plotChartNode=pn.csaPlotChartNode,
            highSeries=pn.csaPlotSeriesMaxCSA,
            lowSeries=pn.csaPlotSeriesMinCSA,
            sliceSeries=pn.csaPlotSeriesCurSlice,
            dynSeries=pn.csaPlotSeriesDynCSA,
        )
        # update progress bar
        progressBar.value = 1
        slicer.app.processEvents()
        # AR
        arDynTable = browser.GetProxyNode(pn.arTableSeq)
        (
            pn.arPlotChartNode,
            pn.arPlotSeriesMaxAR,
            pn.arPlotSeriesMinAR,
            pn.arPlotSeriesCurSlice,
            pn.arPlotSeriesDynAR,
        ) = self.logic.makeARPlot(
            envTable=pn.arEnvTable,
            dynTable=arDynTable,
            sliceIndicatorTable=pn.sliceIndicatorTable,
            plotChartNode=pn.arPlotChartNode,
            highSeries=pn.arPlotSeriesMaxAR,
            lowSeries=pn.arPlotSeriesMinAR,
            sliceSeries=pn.arPlotSeriesCurSlice,
            dynSeries=pn.arPlotSeriesDynAR,
        )
        # update progress bar
        progressBar.value = 2
        slicer.app.processEvents()
        # AR_Axes
        dynTableLong = browser.GetProxyNode(pn.longTableSeq)
        dynTableShort = browser.GetProxyNode(pn.shortTableSeq)
        (
            pn.arAxesPlotChartNode,
            pn.arAxesPlotSeriesLongMax,
            pn.arAxesPlotSeriesLongMin,
            pn.arAxesPlotSeriesShortMax,
            pn.arAxesPlotSeriesShortMin,
            pn.arAxesPlotSeriesCurSlice,
            pn.arAxesPlotSeriesLongDyn,
            pn.arAxesPlotSeriesShortDyn,
        ) = self.logic.makeARAxesPlot(
            envTableLong=pn.longEnvTable,
            dynTableLong=dynTableLong,
            envTableShort=pn.shortEnvTable,
            dynTableShort=dynTableShort,
            sliceIndicatorTable=pn.sliceIndicatorTable,
            plotChartNode=pn.arAxesPlotChartNode,
        )
        # update progress bar
        progressBar.value = 3
        slicer.app.processEvents()
        # Volume (not dynamic, TODO:could have current time marked with indicator)
        pn.volPlotChartNode = self.logic.makeAirVolumePlot(
            tableNode=pn.volTable, plotChartNode=pn.volPlotChartNode
        )
        # TODO: Consider adding vs time plots?
        # CSA/AR/LA/SA vs time for a specific SliceIdx
        # To handle same as the vs dist plots, we would need
        # to have a different browser node for SliceIdx.  It
        # is probably better to have a table that we update on
        # slice index change, or a plotSeries that we update the
        # referred column of a static table.

        # close progress bar
        progressBar.value = nPlots
        slicer.app.processEvents()

    def createOrUpdateSliceIndicatorTable(self, quantData=None, outputTable=None):
        """a table which handles the current slice indicator data
        (vertical lines on all plots at the current slice index).
        The x column just needs to be the current carina distance
        (repeated twice), while the y columns should have the values which
        should be the maximum and minimum y values for the indicator
        vertical line on each plot. For envelope plots, this could be
        min to max for current slice (fancy) or minmin to maxmax.
        For AR, this should always be 0 to 1.
        For ArAxes, this should be short-axis minmin to
        long-axis maxmax.
        """
        pn = self._parameterNode
        scene = slicer.mrmlScene
        if outputTable is None:
            outputTable = scene.AddNewNodeByClass("vtkMRMLTableNode")
            uname = scene.GenerateUniqueName("SliceIndicatorRange")
            outputTable.SetName(uname)
        # Get ranges for the relevant tables
        if quantData is None:
            quantData = self.quantData
        csaRangeCol, arRangeCol, axesRangeCol = (
            self.logic.createSliceIndicatorTableColumns(quantData)
        )
        curCarinaDist = self.getCurrentCarinaDistance()
        distCol = self.logic.createCarinaDistIndicatorTableColumn(curCarinaDist)
        # Actually build the table by adding all the columns
        outputTable.RemoveAllColumns()
        cols = (distCol, csaRangeCol, arRangeCol, axesRangeCol)
        for col in cols:
            outputTable.AddColumn(col)
        return outputTable

    def updateCrossSectionModel(self, quantData=None):
        """Wrapper for logic.updateCrossSectionModel which gathers the inputs
        from the parameter node
        """
        pn = self._parameterNode
        if quantData is None:
            quantData = self.quantData
        if quantData is None:
            logging.warning("No quantData available for updating cross section model!")
            return
        sliceIdx = int(pn.sliceSliderIdx)
        slicePoint = quantData["slicePoints"][sliceIdx, :]
        sliceNormal = quantData["sliceNormals"][sliceIdx, :]
        airSeg = self.getProxyNode(pn.q_airwaySegmentationSequence)
        segmentIdx = pn.q_segmentIdx
        pn.crossSectionModel = self.logic.updateCrossSectionModel(
            slicePoint,
            sliceNormal,
            airSeg,
            segmentIdx=segmentIdx,
            outputModel=pn.crossSectionModel,
        )

    def updateArAxesLines(self):
        """Wrapper for logic.updateArAxisLine which gathers the inputs.
        Creates and styles the markups nodes if needed. Returns immediately
        if quantification isn't run yet.
        """
        pn = self._parameterNode
        if not self.quantData:
            logging.warning(
                "No quantData available for AR axis line updates, cancelling!"
            )
            return
        sliceIdx = int(pn.sliceSliderIdx)
        frameIdx = self.getBrowserNode().GetSelectedItemNumber()
        quantData = self.quantData
        (long1, long2, short1, short2) = self.logic.getArAxisPoints(
            quantData, frameIdx, sliceIdx
        )
        if not pn.longAxisLine:
            pn.longAxisLine = self.logic.initializeArAxisLine(
                "LongAxis", color=self.logic.AR_LONG_AXIS_COLOR
            )
        self.logic.updateArAxisLine(long1, long2, markupsCurve=pn.longAxisLine)
        if not pn.shortAxisLine:
            pn.shortAxisLine = self.logic.initializeArAxisLine(
                "ShortAxis", color=self.logic.AR_SHORT_AXIS_COLOR
            )
        self.logic.updateArAxisLine(short1, short2, markupsCurve=pn.shortAxisLine)

    def initializeReviewSection(self):
        """On completion of quantification run, the "Review Phase" section
        should be initialized by:
        * Set the slice index slider to 0 and set the range
        * link the time index slider to the browser node
        * link the slice index slider callback to update
            * RAS and normal text
            * slice index marker on plots
            * slice index display in 3D view
        """
        pn = self._parameterNode
        scene = slicer.mrmlScene
        quantData = self.quantData
        nSlices = quantData["slicePoints"].shape[0]
        # Set slice slider maximum value
        self.ui.sliceSlider.maximum = nSlices - 1
        # Jump to a middle slice
        self.ui.sliceSlider.value = np.round(nSlices / 2)
        # Create review layout (3D, 3D, cross-section slice, plot)
        self.logic.switchToReviewLayout()
        # Choose plots to show
        plotNodes = [
            pn.csaPlotChartNode,
            pn.arPlotChartNode,
            pn.arAxesPlotChartNode,
            pn.volPlotChartNode,
        ]
        plotViewNodes = list(scene.GetNodesByClass("vtkMRMLPlotViewNode"))
        if len(plotViewNodes) > len(plotNodes):
            # We will run out of plot nodes before view nodes, shorten the list
            plotViewNodes = plotViewNodes[: (len(plotNodes))]
        for idx, plotViewNode in enumerate(plotViewNodes):
            plotViewNode.SetPlotChartNodeID(plotNodes[idx].GetID())
        # Make the quantified airway transparent
        self.logic.setAirwayOpacity(
            opacity=0.5, segSeqOrNode=pn.q_airwaySegmentationSequence
        )
        # Ensure the browser is observed
        self.updateBrowserObservation()

    def onFrameChange(self, unused1, unused2):
        """Called when review phase has been intialized and
        the temporal frame index changes. The slice-index
        based information needs to be updated whenever
        the frame index changes.
        All other frame-based updates (like dynamic plots)
        are handled automatically because they are based on
        sequences that update from the browser already.
        """
        pn = self._parameterNode
        if not self.quantData:
            # no results yet, skip update
            return
        # Pretend the slice index changed to trigger updates
        self.onSliceSliderValueChange(pn.sliceSliderIdx)
        # NOTE: currently this is triggered LOTS of times
        # for each actual frame change. Might be good to
        # do some checking to see whether the update is actually
        # needed or not (e.g. if the previous fully updated combination
        # of slice and frame idxs match the requested ones, then
        # skip the update...) TODO

    def onApplyYMaxCSAButtonClick(self):
        """Set CSA plot y range to be zero to indicated maximum"""
        pn = self._parameterNode
        if not pn.csaPlotChartNode:
            # No CSA plot to update
            return
        yRange = [0, pn.maxY_CSA]
        self.logic.setPlotChartYRange(pn.csaPlotChartNode, yRange)

    def onApplyXRangeButtonClick(self):
        """Apply the chosen plotting range for X"""
        pn = self._parameterNode
        self.updateDistanceAxisLimits(
            pn.carinaDistPlotRangeMm.minimum, pn.carinaDistPlotRangeMm.maximum
        )

    def onCarinaDistPlotRangeSliderChange(self, newMin, newMax):
        """NO LONGER USED
        *** Updated code uses apply button for x range update
        rather than dynamic slider-based updates ***
        Update the cutoff range of carina distance values plotted in plot series"""
        pn = self._parameterNode
        if not self.quantData:
            return
        if pn.updatingCarinaPlotRange:
            return
        # Avoid recursion
        prior = pn.updatingCarinaPlotRange
        pn.updatingCarinaPlotRange = True
        # Perform updates
        pn.carinaDistPlotRangeMm.setRange(newMin, newMax)
        self.updateDistanceAxisLimits(newMin, newMax)
        # self.updateTrimmedTables(newMinDist, newMaxDist)
        slicer.app.processEvents()
        # Done
        pn.updatingCarinaPlotRange = prior  # should this be prior value or just False?

    def updateDistanceAxisLimits(self, newMinDist, newMaxDist, flipAxis=True):
        """Update the range of x values plotted"""
        pn = self._parameterNode
        newXRange = [newMaxDist, newMinDist] if flipAxis else [newMinDist, newMaxDist]
        plotsToUpdate = [
            p
            for p in [pn.arAxesPlotChartNode, pn.csaPlotChartNode, pn.arPlotChartNode]
            if p
        ]
        for plotChartNode in plotsToUpdate:
            self.logic.setPlotChartXRange(plotChartNode, newXRange)

    def updateCSAYAxisLimit(self, newMax):
        """Update the y-axis maximum value for the CSA plot chart"""
        pn = self._parameterNode
        self.logic.setPlotChartYRange(pn.csaPlotChartNode, [0, newMax])

    def onSliceSliderValueChange(self, newSliceIdx):
        """Update the cross-section view model
        in 3D, as well as the aspect ratio axis lines
        in 3D.  Also the CrossSection slice location
        and orientation.
        """
        pn = self._parameterNode
        if not self.quantData:
            return
        if pn.updatingSliceIdx:
            return
        # Avoid recursion
        prior = pn.updatingSliceIdx
        pn.updatingSliceIdx = True
        # Perform updates
        pn.sliceSliderIdx = newSliceIdx

        self.updateCrossSectionModel()
        self.updateArAxesLines()
        self.updateSliceTextFields()
        self.updateSliceIndicatorTableCarinaDistance()
        slicer.app.processEvents()
        # Done
        pn.updatingSliceIdx = prior  # should this be prior value or just False?

    def updateSliceIndicatorTableCarinaDistance(self):
        """Wrapper for logic.updateSliceIndicatorTableCarinaDistance"""
        pn = self._parameterNode
        curCarinaDist = self.getCurrentCarinaDistance()
        self.logic.updateSliceIndicatorTableCarinaDistance(
            curCarinaDist, pn.sliceIndicatorTable
        )

    def updateSliceTextFields(self):
        """Update the displayed text for the centerline slice
        point location and normal direction.
        """
        pn = self._parameterNode
        sliceIdx = int(pn.sliceSliderIdx)
        quantData = self.quantData
        if not quantData:
            self.ui.rasText.text = "(run quantification first)"
            self.ui.normalText.text = "(run quantification first)"
            return
        slicePoint = quantData["slicePoints"][sliceIdx, :]
        R, A, S = (*slicePoint,)
        self.ui.rasText.text = f"({R:0.2f}, {A:0.2f}, {S:0.2f})"
        sliceNormal = quantData["sliceNormals"][sliceIdx, :]
        nR, nA, nS = (*sliceNormal,)
        self.ui.normalText.text = f"({nR:0.2f}, {nA:0.2f}, {nS:0.2f})"
        carinaDist = self.getCurrentCarinaDistance()
        self.ui.carinaDistText.text = f"{carinaDist:0.1f} mm"

    def getCarinaLocation(self):
        """Wrapper for logic.getCarinaLocation"""
        pn = self._parameterNode
        carinaLoc = self.logic.getCarinaLocation(
            pn.carinaLocationsSeqNode,
            pn.currentlyAligned,
            pn.carinaAlignmentTransformSeq,
        )
        return carinaLoc

    def onCenterlineInputSurfaceSelectionChange(self, newNode):
        """This is called whenever the centerline input surface is changed
        in the GUI. If the input surface is a model node, then the segment
        selector widget should be hidden.
        NOTE: this does not cover the case where the parameter node is
        modified through some other method than the GUI!
        """
        pn = self._parameterNode
        pn.cline_InputSurfaceNode = newNode
        if newNode and newNode.IsA("vtkMRMLSegmentationNode"):
            self.ui.centerlineSegmentSelectorWidget.setCurrentSegmentID(
                pn.cline_InputSegmentID
            )
            self.ui.centerlineSegmentSelectorWidget.setVisible(True)
        else:
            self.ui.centerlineSegmentSelectorWidget.setVisible(False)

    def onCenterlineSegmentSelectionChange(self, newSegmentID):
        pn = self._parameterNode
        pn.cline_InputSegmentID = newSegmentID

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

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        print("entering...")
        self.initializeParameterNode()
        print("entered, after init param node...")

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )

    def onStartSceneLoading(self, caller, event):
        if self.parent.isEntered:
            logging.debug(
                "Unobserving parameter node onStartSceneLoading (parent.isEntered)"
            )
            self.setParameterNode(None)
            # self.removeObserver(
            #    self._parameterNode,
            #    vtk.vtkCommand.ModifiedEvent,
            #    self.updateGUIFromParameterNode,
            # )
            # self.initializeParameterNode()
        logging.debug("%s" % event)

    def onSceneLoaded(self, caller, event):
        """Called when a new scene is loaded while the module is open.  Without this, the parameter node is not
        updated with any new default values (test this).
        """
        logging.debug("onSceneLoaded")
        if self.parent.isEntered:
            logging.debug("onSceneLoaded and parent.isEntered")
            self.initializeParameterNode()

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
        # Many methods expect quantData to be stored in the widget,
        # but this doesn't happen automatically on loading parameter node,
        # this should fix that
        self.quantData = self.getQuantDataFromJson()
        if self.quantData:
            self.initializeReviewSection()

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
            # The observer below is triggered whenever the parameter node is modified,
            # it can be thought of a bit like updateGuiFromParameterNode() except
            # that _checkCanApply just handles enabling and disabling GUI elements and
            # modifying toolTips. When one parameter value should update another, that
            # is currently handled via individual callbacks, like
            # onOrigImageSequenceSelectionChange setting the browserNode
            self.addObserver(
                self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply
            )

            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        """Enable or disable buttons based on whether their required inputs
        are present
        """
        pn = self._parameterNode
        # OK this function is called whenever the parameter node is modified
        # It should be safe to use it to control plot or model visibility
        self.updateVisiblity()

        return  # TODO: do some enabling and collapsing here...
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

    def updateVisiblity(self):
        """Make plot item visibility consistent with parameter node values. Also
        control the current slice visibility and AR lines visibility
        """
        pn = self._parameterNode
        # Skip if no plots yet
        if not pn.csaPlotChartNode or not pn.csaPlotSeriesDynCSA:
            return
        # If dynamic curves are hidden, then should envelope curves be solid lines?
        csaCurvesToShow = [
            pn.csaPlotSeriesMaxCSA,
            pn.csaPlotSeriesMinCSA,
            pn.csaPlotSeriesCurSlice,
            pn.csaPlotSeriesDynCSA,
        ]
        arCurvesToShow = [
            pn.arPlotSeriesMaxAR,
            pn.arPlotSeriesMinAR,
            pn.arPlotSeriesCurSlice,
            pn.arPlotSeriesDynAR,
        ]
        arAxesCurvesToShow = [
            pn.arAxesPlotSeriesLongMax,
            pn.arAxesPlotSeriesLongMin,
            pn.arAxesPlotSeriesShortMax,
            pn.arAxesPlotSeriesShortMin,
            pn.arAxesPlotSeriesCurSlice,
            pn.arAxesPlotSeriesLongDyn,
            pn.arAxesPlotSeriesShortDyn,
        ]
        if not pn.showEnvCurvesFlag:
            csaCurvesToShow.remove(pn.csaPlotSeriesMaxCSA)
            csaCurvesToShow.remove(pn.csaPlotSeriesMinCSA)
            arCurvesToShow.remove(pn.arPlotSeriesMaxAR)
            arCurvesToShow.remove(pn.arPlotSeriesMinAR)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesLongMax)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesLongMin)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesShortMax)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesShortMin)
        else:
            # Show them dashed or solid depending on whether the current curve is shown
            if pn.showCurrentCurveFlag:
                envLineStyle = vtkMRMLPlotSeriesNode.LineStyleDash
            else:
                envLineStyle = vtkMRMLPlotSeriesNode.LineStyleSolid
            envCurves = [
                pn.csaPlotSeriesMaxCSA,
                pn.csaPlotSeriesMinCSA,
                pn.arPlotSeriesMaxAR,
                pn.arPlotSeriesMinAR,
                pn.arAxesPlotSeriesLongMax,
                pn.arAxesPlotSeriesLongMin,
                pn.arAxesPlotSeriesShortMax,
                pn.arAxesPlotSeriesShortMin,
            ]
            for envCurve in envCurves:
                envCurve.SetLineStyle(envLineStyle)
        if not pn.showCurrentCurveFlag:
            # Hide dynamic curves
            csaCurvesToShow.remove(pn.csaPlotSeriesDynCSA)
            arCurvesToShow.remove(pn.arPlotSeriesDynAR)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesLongDyn)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesShortDyn)
        if not pn.showSliceIndicatorOnPlotsFlag:
            csaCurvesToShow.remove(pn.csaPlotSeriesCurSlice)
            arCurvesToShow.remove(pn.arPlotSeriesCurSlice)
            arAxesCurvesToShow.remove(pn.arAxesPlotSeriesCurSlice)
        if (
            not pn.showEnvCurvesFlag
            and not pn.showSliceIndicatorOnPlotsFlag
            and not pn.showCurrentCurveFlag
        ):
            # If everything is unchecked, show the dynamic curve to avoid empty plots
            csaCurvesToShow = [pn.csaPlotSeriesDynCSA]
            arCurvesToShow = [pn.arPlotSeriesDynAR]
            arAxesCurvesToShow = [
                pn.arAxesPlotSeriesLongDyn,
                pn.arAxesPlotSeriesShortDyn,
            ]
        # Legend visibility
        for plotChart in [
            pn.csaPlotChartNode,
            pn.arPlotChartNode,
            pn.arAxesPlotChartNode,
            pn.volPlotChartNode,
        ]:
            plotChart.SetLegendVisibility(pn.showLegendFlag)
        # Rebuild plots
        pn.csaPlotChartNode.RemoveAllPlotSeriesNodeIDs()
        for series in csaCurvesToShow:
            pn.csaPlotChartNode.AddAndObservePlotSeriesNodeID(series.GetID())
        pn.arPlotChartNode.RemoveAllPlotSeriesNodeIDs()
        for series in arCurvesToShow:
            pn.arPlotChartNode.AddAndObservePlotSeriesNodeID(series.GetID())
        pn.arAxesPlotChartNode.RemoveAllPlotSeriesNodeIDs()
        for series in arAxesCurvesToShow:
            pn.arAxesPlotChartNode.AddAndObservePlotSeriesNodeID(series.GetID())

        # Current slice cross section model visibility
        pn.crossSectionModel.GetDisplayNode().SetVisibility(pn.showCurrentSlice3DFlag)
        # Axis line visibility in 3D
        pn.longAxisLine.GetDisplayNode().SetVisibility(pn.showARLinesFlag)
        pn.shortAxisLine.GetDisplayNode().SetVisibility(pn.showARLinesFlag)


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
        # CAPITALIZED CONSTANTS
        # self.INIT_CARINA_POINT_COLOR = (1.0, 0.5, 0)  # orange
        self.INIT_CROP_BOX_COLOR = (1.0, 0.75, 0)  # yellow-orange?
        self.INIT_MINIP_SEG_COLOR = self.INIT_CROP_BOX_COLOR
        self.INIT_SEG_SEQ_COLOR = (0.8, 0.35, 0)  # burnt-orange
        self.ALIGNED_CROP_BOX_COLOR = (1.0, 1.0, 0)  # bright yellow
        self.ALIGNED_MINIP_SEG_COLOR = (1.0, 0.9, 0)  # middle yellow
        self.ALIGNED_SEG_SEQ_COLOR = (0.5, 0.68, 0.5)  # slicer default green
        self.VOLUME_QUANT_ROI_COLOR = (0, 0, 1.0)  # blue
        self.RAW_CENTERLINE_COLOR = (0.5, 0.0, 0.0)  # dark red
        self.SMOOTH_CENTERLINE_COLOR = (0.9, 0.0, 0.0)  # brighter red
        self.AIR_THRESH_MIN = -1100
        self.AIR_THRESH_MAX = -250
        self.GROW_MARGIN_MM = 2  # for initializing alignedMinIP seg
        self.CROSS_SECTION_COLOR = (1, 0, 0)  # bright red
        self.AR_LONG_AXIS_COLOR = (1.0, 1.0, 0)  # bright yellow
        self.AR_SHORT_AXIS_COLOR = (1.0, 0.9, 0)  # middle yellow
        self.ENV_CURVE_COLOR = (0.75, 0.75, 0.75)  # light gray
        self.ENV_LINE_STYLE = vtkMRMLPlotSeriesNode.LineStyleDash
        self.ENV_LINE_WIDTH = 2
        self.DYN_CURVE_COLOR = (0.0, 0.0, 0.0)  # black
        self.DYN_LINE_WIDTH = 3
        self.LONG_COLOR = (0.0, 0.0, 0.7)  # dull blue?
        self.ENV_LONG_COLOR = (0.0, 0.0, 0.6)  # darker blue
        self.SHORT_COLOR = (0.0, 0.7, 0.0)  # dull green?
        self.ENV_SHORT_COLOR = (0.0, 0.6, 0.0)  # darker green
        self.SLICE_INDICATOR_COLOR = (
            0.9,
            0,
            0,
        )  # pretty bright red (to match 3D cross section?)
        self.SLICE_INDICATOR_LINE_STYLE = vtkMRMLPlotSeriesNode.LineStyleSolid
        self.SLICE_INDICATOR_LINE_WIDTH = 1
        self.AIR_VOL_SEGMENT_COLOR = (0, 0, 0.75)  # dark blue
        #

    def getParameterNode(self):
        return DynamicMalaciaToolsParameterNode(super().getParameterNode())

    def center3DView(self, viewIdx=0):
        """Center 3D view (like clicking the center button)"""
        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(viewIdx)
        threeDView = threeDWidget.threeDView()
        threeDView.rotateToViewAxis(3)  # look from anterior direction
        threeDView.resetFocalPoint()  # reset the 3D view cube size and center it
        threeDView.resetCamera()  # reset camera zoom

    def setAirwayOpacity(self, opacity, segSeqOrNode):
        """Set opacity for segmentation node, or for the proxy node
        of a supplied sequence node holding segmentations.
        """
        if type(segSeqOrNode) == vtkMRMLSequenceNode:
            seqNode = segSeqOrNode
            segNode = getFirstBrowser(seqNode).GetProxyNode(seqNode)
        else:
            segNode = segSeqOrNode
        dn = segNode.GetDisplayNode()
        dn.SetOpacity(opacity)

    def makeAirVolumePlot(self, tableNode, plotChartNode=None):
        """ """
        scene = slicer.mrmlScene
        if not plotChartNode:
            plotChartNode = scene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            uname = scene.GenerateUniqueName("AirVol_PlotChartNode")
            plotChartNode.SetName(uname)
        # Remove any existing series from the scene (to avoid clutter)
        while plotChartNode.GetPlotSeriesNode():
            oldSeries = plotChartNode.GetPlotSeriesNode()
            # Remove from chart
            plotChartNode.RemoveNthPlotSeriesNodeID(0)
            # Remove from scene
            scene.RemoveNode(oldSeries)
        # Make the new plot series
        volSeries = self.makePlotSeries(
            tableNode=tableNode,
            xColName="TimeSec",
            yColName="Volume_mm3",
            plotSeriesName="AirVolume mm^3",
            lineWidth=3,
        )
        plotChartNode.AddAndObservePlotSeriesNodeID(volSeries.GetID())
        plotChartNode.SetYAxisTitle("Air Volume (mm^3)")
        plotChartNode.SetXAxisTitle("Time (s)")
        plotChartNode.SetTitle("Airway Volume vs Time")
        return plotChartNode

    def makeCSAPlot(
        self,
        envTable,
        dynTable,
        sliceIndicatorTable,
        plotChartNode=None,
        lowSeries=None,
        highSeries=None,
        sliceSeries=None,
        dynSeries=None,
    ):
        """Make (or update) a dynamic plot node for CSA.
        This involves creating several plotSeriesNodes:
         * low, high envelopes (dist vs data)
         * sliceIndicator (vertical line)
         * dynamic curve (dist vs data, linked to proxy node of table sequence)
        """
        scene = slicer.mrmlScene
        if not plotChartNode:
            plotChartNode = scene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            uname = scene.GenerateUniqueName("CSA_PlotChartNode")
            plotChartNode.SetName(uname)
        # Remove any existing series from the scene (to avoid clutter)
        while plotChartNode.GetPlotSeriesNode():
            oldSeries = plotChartNode.GetPlotSeriesNode()
            # Remove from chart
            plotChartNode.RemoveNthPlotSeriesNodeID(0)
            # Remove from scene
            scene.RemoveNode(oldSeries)
        # Make all the new plot series
        lowSeries = self.makePlotSeries(
            tableNode=envTable,
            xColName="DistToCarinaMm",
            yColName="MinCSA",
            plotSeriesName="MinCSA",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_CURVE_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
            outputPlotSeries=lowSeries,
        )
        highSeries = self.makePlotSeries(
            tableNode=envTable,
            xColName="DistToCarinaMm",
            yColName="MaxCSA",
            plotSeriesName="MaxCSA",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_CURVE_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
            outputPlotSeries=highSeries,
        )
        dynSeries = self.makePlotSeries(
            tableNode=dynTable,
            xColName="DistToCarinaMm",
            yColName="CSA",
            plotSeriesName="CSA",
            color=self.DYN_CURVE_COLOR,
            lineWidth=self.DYN_LINE_WIDTH,
            outputPlotSeries=dynSeries,
        )
        sliceSeries = self.makePlotSeries(
            tableNode=sliceIndicatorTable,
            xColName="CurrentSlice",
            yColName="CSA_Range",
            plotSeriesName="CurrentSlice",
            color=self.SLICE_INDICATOR_COLOR,
            lineStyle=self.SLICE_INDICATOR_LINE_STYLE,
            lineWidth=self.SLICE_INDICATOR_LINE_WIDTH,
            outputPlotSeries=sliceSeries,
        )
        # TODO: consider adding a series which puts a marker
        # at the point with the lowest minimum CSA... (could
        # be added to sliceIndicatorTable for data storage)
        seriesList = [highSeries, lowSeries, sliceSeries, dynSeries]
        for series in seriesList:
            plotChartNode.AddAndObservePlotSeriesNodeID(series.GetID())
        # Set up axes labels
        plotChartNode.SetXAxisTitle("Distance From Carina (mm)")
        # Reverse the X axis and explicitly set the range
        distCol = dynTable.GetTable().GetColumnByName("DistToCarinaMm")
        dists = numpy_support.vtk_to_numpy(distCol)
        distMin = np.min(dists)
        distMax = np.max(dists)
        xRange = (distMax, distMin)  # reverse axis
        plotChartNode.SetXAxisRangeAuto(False)
        plotChartNode.SetXAxisRange(*xRange)
        # Y axis
        plotChartNode.SetYAxisTitle("Cross-sectional Area mm^2")
        # TODO: consider adding overall title (SetTitle()),
        plotChartNode.SetTitle("CSA along Airway")
        # grid (SetGridVisibility())
        # legend (SetLegendVisibility()), etc.
        return plotChartNode, highSeries, lowSeries, sliceSeries, dynSeries

    def setPlotChartXRange(self, plotChartNode, newXRange: List[float]):
        """Force a plotting x axis range. Note that if the first
        element of the range is greater than the second, the x-axis
        will be reversed.
        """
        plotChartNode.SetXAxisRangeAuto(False)
        plotChartNode.SetXAxisRange(*newXRange)

    def setPlotChartYRange(self, plotChartNode, newYRange: List[float]):
        """Force a plotting y axis range."""
        plotChartNode.SetYAxisRangeAuto(False)
        plotChartNode.SetYAxisRange(*newYRange)

    def makeARPlot(
        self,
        envTable,
        dynTable,
        sliceIndicatorTable,
        plotChartNode=None,
        highSeries=None,
        lowSeries=None,
        sliceSeries=None,
        dynSeries=None,
    ):
        """Make (or update) a dynamic plot node for AR.
        This involves creating several plotSeriesNodes:
         * low, high envelopes (dist vs data)
         * sliceIndicator (vertical line)
         * dynamic curve (dist vs data, linked to proxy node of table sequence)
        """
        scene = slicer.mrmlScene
        if not plotChartNode:
            plotChartNode = scene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            uname = scene.GenerateUniqueName("AR_PlotChartNode")
            plotChartNode.SetName(uname)
        # Remove any existing series from the scene (to avoid clutter)
        while plotChartNode.GetPlotSeriesNode():
            oldSeries = plotChartNode.GetPlotSeriesNode()
            # Remove from chart
            plotChartNode.RemoveNthPlotSeriesNodeID(0)
            # Remove from scene
            scene.RemoveNode(oldSeries)
        # Make all the new plot series
        lowSeries = self.makePlotSeries(
            tableNode=envTable,
            xColName="DistToCarinaMm",
            yColName="MinAR",
            plotSeriesName="MinAR",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_CURVE_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
            outputPlotSeries=lowSeries,
        )
        highSeries = self.makePlotSeries(
            tableNode=envTable,
            xColName="DistToCarinaMm",
            yColName="MaxAR",
            plotSeriesName="MaxAR",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_CURVE_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
            outputPlotSeries=highSeries,
        )
        dynSeries = self.makePlotSeries(
            tableNode=dynTable,
            xColName="DistToCarinaMm",
            yColName="AR",
            plotSeriesName="AR",
            color=self.DYN_CURVE_COLOR,
            lineWidth=self.DYN_LINE_WIDTH,
            outputPlotSeries=dynSeries,
        )
        sliceSeries = self.makePlotSeries(
            tableNode=sliceIndicatorTable,
            xColName="CurrentSlice",
            yColName="AR_Range",
            plotSeriesName="CurrentSlice",
            color=self.SLICE_INDICATOR_COLOR,
            lineStyle=self.SLICE_INDICATOR_LINE_STYLE,
            lineWidth=self.SLICE_INDICATOR_LINE_WIDTH,
            outputPlotSeries=sliceSeries,
        )
        # TODO: consider adding a series which puts a marker
        # at the point with the lowest minimum CSA... (could
        # be added to sliceIndicatorTable for data storage)
        seriesList = [highSeries, lowSeries, sliceSeries, dynSeries]
        for series in seriesList:
            plotChartNode.AddAndObservePlotSeriesNodeID(series.GetID())
        # Set up axes labels
        plotChartNode.SetXAxisTitle("Distance From Carina (mm)")
        # Reverse the X axis and explicitly set the range
        distCol = dynTable.GetTable().GetColumnByName("DistToCarinaMm")
        dists = numpy_support.vtk_to_numpy(distCol)
        distMin = np.min(dists)
        distMax = np.max(dists)
        xRange = (distMax, distMin)  # reverse axis
        plotChartNode.SetXAxisRangeAuto(False)
        plotChartNode.SetXAxisRange(*xRange)
        # Y axis (force 0 to 1 for aspect ratios)
        plotChartNode.SetYAxisRangeAuto(False)
        plotChartNode.SetYAxisRange(0, 1)
        plotChartNode.SetYAxisTitle("Aspect Ratio (short/long)")
        # TODO: consider adding overall title (SetTitle()),
        plotChartNode.SetTitle("Aspect Ratio along Airway")
        # grid (SetGridVisibility())
        # legend (SetLegendVisibility()), etc.
        return plotChartNode, highSeries, lowSeries, sliceSeries, dynSeries

    def makeARAxesPlot(
        self,
        envTableLong,
        dynTableLong,
        envTableShort,
        dynTableShort,
        sliceIndicatorTable,
        plotChartNode=None,
        highSeriesLong=None,
        lowSeriesLong=None,
        highSeriesShort=None,
        lowSeriesShort=None,
        sliceSeries=None,
        dynSeriesLong=None,
        dynSeriesShort=None,
    ):
        """Make (or update) a dynamic plot node for AR Axes.
        This involves creating several plotSeriesNodes for each axis:
         * low, high envelopes (dist vs data)
         * sliceIndicator (vertical line)
         * dynamic curve (dist vs data, linked to proxy node of table sequence)
        """
        scene = slicer.mrmlScene
        if not plotChartNode:
            plotChartNode = scene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            uname = scene.GenerateUniqueName("ArAxes_PlotChartNode")
            plotChartNode.SetName(uname)
        # Remove any existing series from the scene (to avoid clutter)
        while plotChartNode.GetPlotSeriesNode():
            oldSeries = plotChartNode.GetPlotSeriesNode()
            # Remove from chart
            plotChartNode.RemoveNthPlotSeriesNodeID(0)
            # Remove from scene
            scene.RemoveNode(oldSeries)
        # Make all the new plot series
        lowSeriesLong = self.makePlotSeries(
            tableNode=envTableLong,
            xColName="DistToCarinaMm",
            yColName="MinLongAxisMm",
            plotSeriesName="LongAxisMin",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_LONG_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
        )
        highSeriesLong = self.makePlotSeries(
            tableNode=envTableLong,
            xColName="DistToCarinaMm",
            yColName="MaxLongAxisMm",
            plotSeriesName="LongAxisMax",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_LONG_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
        )
        dynSeriesLong = self.makePlotSeries(
            tableNode=dynTableLong,
            xColName="DistToCarinaMm",
            yColName="LongAxis",
            plotSeriesName="LongAxis",
            color=self.LONG_COLOR,
            lineWidth=self.DYN_LINE_WIDTH,
        )
        lowSeriesShort = self.makePlotSeries(
            tableNode=envTableShort,
            xColName="DistToCarinaMm",
            yColName="MinShortAxisMm",
            plotSeriesName="ShortAxisMin",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_SHORT_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
        )
        highSeriesShort = self.makePlotSeries(
            tableNode=envTableShort,
            xColName="DistToCarinaMm",
            yColName="MaxShortAxisMm",
            plotSeriesName="ShortAxisMax",
            lineStyle=self.ENV_LINE_STYLE,
            color=self.ENV_SHORT_COLOR,
            lineWidth=self.ENV_LINE_WIDTH,
        )
        dynSeriesShort = self.makePlotSeries(
            tableNode=dynTableShort,
            xColName="DistToCarinaMm",
            yColName="ShortAxis",
            plotSeriesName="ShortAxis",
            color=self.SHORT_COLOR,
            lineWidth=self.DYN_LINE_WIDTH,
        )
        sliceSeries = self.makePlotSeries(
            tableNode=sliceIndicatorTable,
            xColName="CurrentSlice",
            yColName="AxesLengthRange",
            plotSeriesName="CurrentSlice",
            color=self.SLICE_INDICATOR_COLOR,
            lineStyle=self.SLICE_INDICATOR_LINE_STYLE,
            lineWidth=self.SLICE_INDICATOR_LINE_WIDTH,
        )
        # TODO: consider adding a series which puts a marker
        # at the point with the lowest minimum CSA... (could
        # be added to sliceIndicatorTable for data storage)
        seriesList = [
            highSeriesLong,
            lowSeriesLong,
            highSeriesShort,
            lowSeriesShort,
            sliceSeries,
            dynSeriesLong,
            dynSeriesShort,
        ]
        for series in seriesList:
            plotChartNode.AddAndObservePlotSeriesNodeID(series.GetID())
        # Set up axes labels
        plotChartNode.SetXAxisTitle("Distance From Carina (mm)")
        # Reverse the X axis and explicitly set the range
        distCol = dynTableLong.GetTable().GetColumnByName("DistToCarinaMm")
        dists = numpy_support.vtk_to_numpy(distCol)
        distMin = np.min(dists)
        distMax = np.max(dists)
        xRange = (distMax, distMin)  # reverse axis
        plotChartNode.SetXAxisRangeAuto(False)
        plotChartNode.SetXAxisRange(*xRange)
        # Y axis
        plotChartNode.SetYAxisTitle("Axis Length (mm)")
        plotChartNode.SetTitle("Long and Short Axis Length along Airway")
        # TODO: consider adding overall title (SetTitle()),
        # grid (SetGridVisibility())
        # legend (SetLegendVisibility()), etc.
        return plotChartNode, *seriesList

    def makePlotSeries(
        self,
        tableNode: vtkMRMLTableNode,
        xColName: str,
        yColName: str,
        plotSeriesName: str,
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        lineWidth: float = 1.0,
        lineStyle: int = vtkMRMLPlotSeriesNode.LineStyleSolid,
        markerStyle: int = vtkMRMLPlotSeriesNode.MarkerStyleNone,
        plotType: int = vtkMRMLPlotSeriesNode.PlotTypeScatter,
        outputPlotSeries=None,
    ):
        """Make a plot series node"""
        scene = slicer.mrmlScene
        if outputPlotSeries is None:
            outputPlotSeries = scene.AddNewNodeByClass("vtkMRMLPlotSeriesNode")
        # The series name shows up on any legends, so it matters more than usual
        outputPlotSeries.SetName(plotSeriesName)
        # Set up series properties
        outputPlotSeries.SetAndObserveTableNodeID(tableNode.GetID())
        outputPlotSeries.SetXColumnName(xColName)
        outputPlotSeries.SetYColumnName(yColName)
        outputPlotSeries.SetPlotType(plotType)
        outputPlotSeries.SetColor(color)
        outputPlotSeries.SetLineWidth(lineWidth)
        outputPlotSeries.SetLineStyle(lineStyle)
        outputPlotSeries.SetMarkerStyle(markerStyle)
        return outputPlotSeries

    def updateSliceIndicatorTableCarinaDistance(
        self, dist, tableNode, colName="CurrentSlice"
    ):
        """Update the carina distance in the slice indicator table
        (likely to reflect new slice index)
        """
        col = tableNode.GetTable().GetColumnByName(colName)
        col.SetValue(0, dist)
        col.SetValue(1, dist)
        # The table node is not automatically updated, we have to
        # trigger it
        tableNode.Modified()

    def getCarinaLocation(
        self, carinaLocSeqNode, currentlyAlignedFlag, alignmentTransformSeq
    ):
        """Get the aligned carina location.  Catch if things are not
        currently aligned."""
        browser = getFirstBrowser(carinaLocSeqNode)
        carinaLocNode = browser.GetProxyNode(carinaLocSeqNode)
        if currentlyAlignedFlag:
            # get the current location of any of the points
            carinaLoc = np.array(carinaLocNode.GetNthControlPointPositionWorld(0))
        else:
            # Need to apply the alignment transform before getting location
            tForm = browser.GetProxyNode(alignmentTransformSeq)
            tempCarinaLocNode = cloneItem(carinaLocNode)
            tempCarinaLocNode.SetAndObserveTransformNodeID(tForm.GetID())
            tempCarinaLocNode.HardenTransform()
            carinaLoc = np.array(tempCarinaLocNode.GetNthControlPointPositionWorld(0))
            slicer.mrmlScene.RemoveNode(tempCarinaLocNode)
        return carinaLoc

    def findCarinaDistances(self, carinaLoc, slicePoints):
        """For each of the slicing center points, find the distance
        from the aligned carina location.
        """
        # Format carina location coordinates so they broadcast appropriately
        carinaLoc = np.array(carinaLoc).reshape((1, -1))
        distances = np.sqrt(np.sum(np.power(carinaLoc - slicePoints, 2), axis=1))
        return distances

    def createOrUpdateQuantDataTextNode(self, quantData, textNode=None):
        """Serialize the quantification output to a text node
        to allow saving/restoring in a persistent way to the scene.
        """
        scene = slicer.mrmlScene
        if textNode is None:
            textNode = scene.AddNewNodeByClass("vtkMRMLTextNode")
            uname = scene.GenerateUniqueName("QuantDataJsonText")
            textNode.SetName(uname)
        quantDataText = serialize_dict_with_ndarrays(quantData)
        textNode.SetText(quantDataText)
        return textNode

    def createReviewLayout(self, layoutIdNumber=555):
        """Create and register quantification review 4-up layout
        3D 1, 3D 2
        plot, plot
        """
        customLayout = """
    <layout type="vertical" split="true">
        <item>
        <layout type="horizontal">
            <item>
                <view class="vtkMRMLViewNode" singletontag="1">
                    <property name="viewlabel" action="default">1</property>
                </view>
            </item>
            <item>
                <view class="vtkMRMLViewNode" singletontag="2">
                    <property name="viewlabel" action="default">2</property>
                </view>
            </item>
        </layout>
        </item>
        <item>
        <layout type="horizontal">
            <item>
                <view class="vtkMRMLPlotViewNode" singletontag="Plot1">
                    <property name="viewlabel" action="default">Plot1</property>
                </view>
            </item>
            <item>
                <view class="vtkMRMLPlotViewNode" singletontag="Plot2">
                    <property name="viewlabel" action="default">Plot2</property>
                </view>
            </item>
        </layout>
        </item>
    </layout>
    """
        # If wanted later for a cross-section-aligned slice node, just
        # didn't want to delete the example.
        sliceNodeAlternate = """<item>
                <view class="vtkMRMLSliceNode" singletontag="CrossSection">
                    <property name="orientation" action="default">Axial</property>
                    <property name="viewlabel" action="default">CrossSection</property>
                    <property name="viewcolor" action="default">#59CBDA</property>
                </view>
            </item> """
        layoutManager = slicer.app.layoutManager()
        layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(
            layoutIdNumber, customLayout
        )

        # Add button to layout selector toolbar for this custom layout (only if not already there)
        viewToolBar = slicer.util.mainWindow().findChild("QToolBar", "ViewToolBar")
        layoutMenu = viewToolBar.widgetForAction(viewToolBar.actions()[0]).menu()
        existingLayoutTextList = [action.text for action in layoutMenu.actions()]
        customLayoutText = "Quantification Review 4-Up"
        if customLayoutText not in existingLayoutTextList:
            layoutSwitchActionParent = layoutMenu  # use `layoutMenu` to add inside layout list, use `viewToolBar` to add next the standard layout list
            layoutSwitchAction = layoutSwitchActionParent.addAction(
                customLayoutText
            )  # add inside layout list
            layoutSwitchAction.setData(layoutIdNumber)
            layoutSwitchAction.setIcon(qt.QIcon(":Icons/LayoutFourUpView.png"))
            layoutSwitchAction.setToolTip("Malacia Review 4-UP")
            layoutSwitchAction.connect(
                "triggered()",
                lambda layoutId=layoutIdNumber: slicer.app.layoutManager().setLayout(
                    layoutId
                ),
            )

    def switchToReviewLayout(self):
        """Switch to custom 4-up layout (3D x2, cross section slice,
        and plot)"""
        slicer.app.layoutManager().setLayout(555)

    def switchToRegularLayout(self):
        """Switch to regular 4-Up layout"""
        slicer.app.layoutManager().setLayout(3)

    def initializeArAxisLine(self, name: str, color=(1.0, 1.0, 0.0)):
        """Initialize a markups curve node to represent one of the
        axes which contribute to the aspect ratio measured
        """
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", name)
        node.RemoveAllControlPoints()
        node.AddControlPoint(0, 0, 0)
        node.AddControlPoint(0, 0, 0)
        # Hide each individual control point (show only line)
        for cpIdx in range(node.GetNumberOfControlPoints()):
            node.SetNthControlPointVisibility(cpIdx, False)
        # Prevent moving points with mouse
        node.SetLocked(True)
        dn = node.GetDisplayNode()
        dn.SetGlyphScale(2.0)
        dn.SetUseGlyphScale(True)
        dn.SetSelectedColor(color)
        return node

    def createCarinaDistIndicatorTableColumn(self, carinaDist):
        """Make carina distance column for slice indicator table.
        (just duplicates carina distance )"""
        colName = "CurrentSlice"
        col = makeFloatCol(colName, np.vstack((carinaDist, carinaDist)))
        return col

    def createSliceIndicatorTableColumns(self, quantData):
        """ """
        csaData = quantData["csaData"]
        csaRangeCol = makeRangeCol("CSA_Range", csaData)
        arData = quantData["arData"]
        arRangeCol = makeRangeCol("AR_Range", arData)
        longAxisData = quantData["longAxisData"]  # frames,slices,7 (length first)
        longAxisLength = np.squeeze(longAxisData[:, :, 0])
        shortAxisData = quantData["shortAxisData"]
        shortAxisLength = np.squeeze(shortAxisData[:, :, 0])
        axesData = np.hstack((longAxisLength, shortAxisLength))
        axesRangeCol = makeRangeCol("AxesLengthRange", axesData)
        return (csaRangeCol, arRangeCol, axesRangeCol)

    def updateArAxisLine(self, p1, p2, markupsCurve):
        """Update the supplied curve to have the supplied points
        as first and second control points."""
        cpArr = np.vstack((p1, p2))
        slicer.util.updateMarkupsControlPointsFromArray(markupsCurve, cpArr)
        for cpIdx in range(markupsCurve.GetNumberOfControlPoints()):
            markupsCurve.SetNthControlPointVisibility(cpIdx, False)

    def updateCrossSectionModel(
        self, slicePoint, sliceNormal, airSeg, segmentIdx=0, outputModel=None
    ):
        """Create or update a cross-section model by re-finding and triangulating
        the clip contour of the airway. The plane defined by the slicePoint and
        sliceNormal is used to clip the segmentIdx'th segment in the airSeg
        segmentation, and the output is put into the outputModel.  If the
        outputModel is None, then a new model is created and returned.
        """
        scene = slicer.mrmlScene
        if outputModel is None:
            outputModel = scene.AddNewNodeByClass("vtkMRMLModelNode")
            name = "CrossSection"
            uname = scene.GenerateUniqueName(name)
            outputModel.SetName(uname)
            # set up diplay properties
            outputModel.CreateDefaultDisplayNodes()
            dn = outputModel.GetDisplayNode()
            dn.SetColor(self.CROSS_SECTION_COLOR)
        # Get segment ID (assume first by default)
        segID = airSeg.GetSegmentation().GetNthSegmentID(segmentIdx)
        polydata = vtk.vtkPolyData()
        airSeg.GetClosedSurfaceRepresentation(segID, polydata)
        surf = pv.wrap(polydata)
        clip = surf.slice(normal=sliceNormal, origin=slicePoint)
        clip_connected = clip.connectivity("closest", closest_point=slicePoint)
        # clip_connected.triangulate_contour()
        triangulator = vtk.vtkContourTriangulator()
        triangulator.SetInputData(clip_connected)
        triangulator.Update()
        crossSectionPolyData = triangulator.GetOutput()
        # Fix the surface normals (all to the slice normal)
        # (if we don't do this, I think the contour points
        # retain their old surface normals, which ends up
        # shading the slice really weirdly)
        pointData = crossSectionPolyData.GetPointData()
        nPoints = pointData.GetNumberOfTuples()
        vtkNormals = vtk.vtkFloatArray()
        vtkNormals.SetNumberOfComponents(3)
        vtkNormals.SetNumberOfTuples(nPoints)
        for idx in range(nPoints):
            # All points should have the same normal, the slice normal
            vtkNormals.SetTuple(idx, sliceNormal)
        pointData.SetNormals(vtkNormals)
        # Make this the model's polydata
        outputModel.SetAndObservePolyData(crossSectionPolyData)
        # And return it
        return outputModel

    def getArAxisPoints(self, quantData, frameIdx, sliceIdx):
        """Get the 4 spatial points defining the long and short
        axis used for calculating the aspect ratio in the given
        frame and slice.
        """
        if quantData is None:
            return None
        longAxisData = quantData["longAxisData"]
        long1 = longAxisData[frameIdx, sliceIdx, 1:4]
        long2 = longAxisData[frameIdx, sliceIdx, 4:7]
        shortAxisData = quantData["shortAxisData"]
        short1 = shortAxisData[frameIdx, sliceIdx, 1:4]
        short2 = shortAxisData[frameIdx, sliceIdx, 4:7]
        return (long1, long2, short1, short2)

    def runQuantification(
        self,
        airSegSeq,
        centerline,
        volQuantROI,
        airSegName="AlignedAirway",
    ):
        """Run the CSA, AR, and volume quantification steps on all frames
        and return quantification data in a dictionary for further processing

        Output data structure is a bit complicated.
        quantData is a list, with one element per frame.
        Each frame element is a tuple, where the second element of
        the tuple is the quantified segment volume in cubic mm.
        The first element of the tuple is a list of dictionaries,
        where each element of the list corresponds to a point
        along the centerline.  For each point, the dictionary
        has keys CSA, AR, Long Axis, and Short Axis.

        Thus, to get the Short Axis length of the 10th centerline
        point (index 9) in the 3rd frame (index 2), you would
        want
        shortAxisLength = quantData[2][0][9]["Short Axis"]
        For volume in the same frame, you would want:
        airVolMm3 = quantData[2][1]

        quantData[frameIdx][1] is volume in that frame
        quantData[frameIdx][0][centerlineIdx][key] is pointwise quantified value
        """
        browser = getFirstBrowser(airSegSeq)
        airSeg = browser.GetProxyNode(airSegSeq)
        # For volume quantification, there is not need to regenerate the
        # ROI segment on every frame, it is the same, so let's just
        # create it once and then re-use it
        volSeg = cloneItem(airSeg)
        volSeg.SetName("TempROISegmentation")
        volSeg.GetSegmentation().RemoveAllSegments()
        volSegmentID = fillROISegment(volQuantROI, volSeg)
        # Set up progress bar
        nFrames = airSegSeq.GetNumberOfDataNodes()
        progressBar = createProgressDialog(
            value=0,
            maximum=nFrames,
            autoClose=True,
            windowTitle="Quantifying CSA, AR, and Volume...",
            labelText=f"Processing frame 0 of {nFrames}",
        )
        slicePoints, sliceNormals = getCenterlineSlicePointsAndNormals(centerline)
        nSlices = sliceNormals.shape[0]
        # Organize output arrays
        # Some are frames by points (CSA, AR, etc)
        csaData = np.ndarray((nFrames, nSlices), dtype=float)
        arData = np.ndarray((nFrames, nSlices), dtype=float)
        # Axis data is frames by points x 7 (length,R,A,S,R,A,S)
        longAxisData = np.ndarray((nFrames, nSlices, 7), dtype=float)
        shortAxisData = np.ndarray((nFrames, nSlices, 7), dtype=float)
        # Some are just frames (vols)
        airwayVolData = np.ndarray((nFrames,), dtype=float)
        # Some are just points (slicePoints, sliceNormals) (each nPoints x 3)
        for frameIdx in range(nFrames):
            browser.SetSelectedItemNumber(frameIdx)
            progressBar.value = frameIdx
            progressBar.labelText = f"Processing frame {frameIdx} of {nFrames-1}..."
            slicer.app.processEvents()
            # Get airway as polydata
            airSegID = airSeg.GetSegmentation().GetSegmentIdBySegmentName(airSegName)
            airPolyData = vtk.vtkPolyData()
            airSeg.GetClosedSurfaceRepresentation(airSegID, airPolyData)
            surf = pv.wrap(airPolyData)
            for sliceIdx, (point, normal) in enumerate(zip(slicePoints, sliceNormals)):
                clip = surf.slice(normal=normal, origin=point)
                if clip.n_points == 0:
                    # Most likely, the slicing plane missed the airway surface!
                    # What should be done in this case?
                    logging.warning(
                        f"Slice {sliceIdx} in frame {frameIdx} generated no points when clipped!"
                    )
                    # For now, just throw an exception because we have no way of handling missing data
                    # in the plots/slice models/long axis/short axis/etc.
                    raise RuntimeError(
                        f"Slice {sliceIdx} in frame {frameIdx} generated no points when clipped!"
                    )
                    # TODO: consider automatically cleaning up volume regions up to this point
                clip_connected = clip.connectivity("closest", closest_point=point)
                arDataDict = compute_aspect_ratio(clip_connected, normal)
                ar = arDataDict["aspectRatio"]
                longAxisLength = arDataDict["longAxisLength"]
                longAxisPt1 = arDataDict["longAxisPoints"][0]
                longAxisPt2 = arDataDict["longAxisPoints"][1]
                shortAxisLength = arDataDict["shortAxisLength"]
                shortAxisPt1 = arDataDict["shortAxisPoints"][0]
                shortAxisPt2 = arDataDict["shortAxisPoints"][1]
                csa = compute_area(clip_connected)
                # Store data in arrays
                csaData[frameIdx, sliceIdx] = csa
                arData[frameIdx, sliceIdx] = ar
                longAxisData[frameIdx, sliceIdx, :] = (
                    longAxisLength,
                    *longAxisPt1,
                    *longAxisPt2,
                )
                shortAxisData[frameIdx, sliceIdx, :] = (
                    shortAxisLength,
                    *shortAxisPt1,
                    *shortAxisPt2,
                )
            # Find air volume in frame
            frameVolumeMm3 = self.findAirwayVolumeInBoxMm3(
                airSegID, airSeg, volSegmentID, volSeg
            )
            airwayVolData[frameIdx] = frameVolumeMm3

        # Close progress bar
        progressBar.value = nFrames
        # Remove the filled ROI segmentation
        slicer.mrmlScene.RemoveNode(volSeg)
        # Bundle the output data
        quantData = {
            "csaData": csaData,
            "arData": arData,
            "longAxisData": longAxisData,
            "shortAxisData": shortAxisData,
            "airwayVolMm3Data": airwayVolData,
            "slicePoints": slicePoints,
            "sliceNormals": sliceNormals,
        }
        return quantData

    def findAirwayVolumeInBoxMm3(
        self,
        airwaySegID,
        airSegNode,
        volBoxSegID,
        volBoxSegNode,
        keepIntersectedSegment=True,
    ) -> float:
        """Finds the volume of the intersection between two segments in mm^3.
        THe first should represent an airway, and the second should have been
        generated from a box ROI (for example by fillROISegment).

        The intersected segment is removed if keepIntersectedSegment is False,
        otherwise, it is retained.  Either way, the volBoxSeg is not retained
        in the airway segmentation.
        """
        # Duplicate airway segment
        airVolSegID = copy_segment("AirwayVol", airSegNode, airwaySegID)
        # Set the color (to avoid rainbow segments across frames!)
        airVolSegment = airSegNode.GetSegmentation().GetSegment(airVolSegID)
        airVolSegment.SetColor(*self.AIR_VOL_SEGMENT_COLOR)
        # Get real name (might be changed to be unique)
        airVolSegName = airVolSegment.GetName()
        # Intersect it with the volume quant box segment
        self.limitSegmentRegion(airVolSegID, airSegNode, volBoxSegID, volBoxSegNode)
        # Find the limited segment volume
        segStatsLogic = self.generateSegmentVolumeStats(airSegNode)
        volumeMeasurementNameMm3 = "LabelmapSegmentStatisticsPlugin.volume_mm3"
        segStats = segStatsLogic.getStatistics()
        volMm3 = segStats[airVolSegName, volumeMeasurementNameMm3]
        if not keepIntersectedSegment:
            airSegNode.RemoveSegment(airVolSegID)
        return volMm3

    def getCarinaTransforms(self, carinaSeq, outputTransformSeq=None):
        """Create or update a sequence of linear, translation-only transforms
        which move each of the original carina landmarks to the same location.
        """
        scene = slicer.mrmlScene
        browser = getFirstBrowser(carinaSeq)
        outputTransformSeq = getTransformSequenceFromLandmarksSequence(
            carinaSeq, outputTransformSeq, addToBrowser=False
        )
        outputTransformSeq.SetAttribute(
            "DataNodeClassName", "vtkMRMLLinearTransformNode"
        )
        # Ensure sequence is added to browser node (done here to customize more)
        seqID = outputTransformSeq.GetID()
        if not browser.IsSynchronizedSequenceNodeID(seqID):
            browser.AddSynchronizedSequenceNodeID(seqID)
        # Standardize settings and proxy node name
        browser.SetSaveChanges(outputTransformSeq, True)
        browser.SetOverwriteProxyName(outputTransformSeq, False)
        segProxy = browser.GetProxyNode(outputTransformSeq)
        name = f"AlignmentTransform"
        uname = scene.GenerateUniqueName(name)
        segProxy.SetName(uname)
        return outputTransformSeq

    def applyTransformToProxies(
        self, seqList: List[vtkMRMLSequenceNode], transformSeq: vtkMRMLSequenceNode
    ):
        """Soft-apply the proxy transform to the proxy node of each of the sequences
        in seqList.
        """
        browser = getFirstBrowser(transformSeq)
        tForm = browser.GetProxyNode(transformSeq)
        for seq in seqList:
            prox = browser.GetProxyNode(seq)
            prox.SetAndObserveTransformNodeID(tForm.GetID())

    def unapplyTransformsOnProxies(self, seqList: List[vtkMRMLSequenceNode]):
        """Remove any observed parent transform on the proxy node of each
        of the sequences in seq
        """
        browser = getFirstBrowser(seqList[0])
        for seq in seqList:
            prox = browser.GetProxyNode(seq)
            prox.SetAndObserveTransformNodeID(None)

    def setItemsVisibility(self, items: List, showFlag: bool = False) -> None:
        """Show/Hide the items in the list. If list items are sequences
        the visibility is set on the proxy node of the first browser.
        """
        for item in items:
            # Skip empty items
            if not item:
                continue
            # Handle sequences and non-sequences
            if isinstance(item, vtkMRMLSequenceNode):
                prox = getFirstBrowser(item).GetProxyNode(item)
                prox.GetDisplayNode().SetVisibility(showFlag)
            else:
                item.GetDisplayNode().SetVisibility(showFlag)

    def transferSegmentationToAlignedMinIP(
        self, initialAirSegSeq, alignedMinIP, alignedMinIPSegmentation=None
    ) -> Tuple[vtkMRMLSegmentationNode, str]:
        """Initialize the aligned minIP segmentation from the initial segmentation.
        Specifically, this is achieved by transfering the current frame initial airway segmentation to the
        aligned minIP, then growing that by a defined margin, then trimming via threshold, then keeping
        largest island. (order is actually implemented differently, but same effect)
        If initial segmentation is poor, it might be helpful to choose a good frame
        to start the transfer from, but the AlignedMinIP is probably almost always going
        to need manual editing.
        """
        scene = slicer.mrmlScene
        browser = getFirstBrowser(initialAirSegSeq)
        initAirSeg = browser.GetProxyNode(initialAirSegSeq)
        if not alignedMinIPSegmentation:
            alignedMinIPSegmentation = scene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode"
            )
            name = "AlignedMinIPSegmentation"
            uname = scene.GenerateUniqueName(name)
            alignedMinIPSegmentation.SetName(uname)
        # Clear out the minIP segmentation to start fresh
        alignedMinIPSegmentation.GetSegmentation().RemoveAllSegments()
        # All air by initial threshold
        threshSegID = self.segmentByThreshold(
            alignedMinIPSegmentation,
            alignedMinIP,
            "AlignedMinIP_Airway",
            thresholdMinValue=self.AIR_THRESH_MIN,
            thresholdMaxValue=self.AIR_THRESH_MAX,
        )
        # Copy initial segment ( Assumes airway is first segment...)
        initSegID = initAirSeg.GetSegmentation().GetNthSegmentID(0)
        alignedMinIPSegmentation.GetSegmentation().CopySegmentFromSegmentation(
            initAirSeg.GetSegmentation(), initSegID
        )
        copiedSegID = alignedMinIPSegmentation.GetSegmentation().GetNthSegmentID(1)
        # Grow the segment
        self.segmentMarginAdjust(
            alignedMinIPSegmentation,
            alignedMinIP,
            copiedSegID,
            marginMm=self.GROW_MARGIN_MM,
        )
        # Intersect the all-air segment with the copied-then-grown segment
        intersect_segment(threshSegID, copiedSegID, alignedMinIPSegmentation)
        # Keep only the largest island (remove anything disconnected over a border)
        keep_largest_island(threshSegID, alignedMinIPSegmentation)
        # Set segment color
        segment = alignedMinIPSegmentation.GetSegmentation().GetSegment(threshSegID)
        segment.SetColor(self.ALIGNED_MINIP_SEG_COLOR)
        # Remove the extra segment (the copied/grown one)
        alignedMinIPSegmentation.GetSegmentation().RemoveSegment(copiedSegID)
        # Generate 3D representation
        alignedMinIPSegmentation.CreateClosedSurfaceRepresentation()
        # Restore the frame index we were originally on
        # browser.SetSelectedItemNumber(currentFrameIdx)
        return alignedMinIPSegmentation, threshSegID

    def initializeRegistrationPhase(
        self,
        origImageSequenceNode=None,
        browserNode=None,
        initialCarinaLandmark=None,
        initialCropBox=None,
        nonLinAlignTformSeq1=None,
        nonLinAlignTformSeq2=None,
        carinaAlignmentTransformSeq=None,
    ):
        """Handle any needed creation of nodes for registratin phase, as well
        as standardizing colors, naming, etc.
        """
        scene = slicer.mrmlScene
        if not origImageSequenceNode:
            slicer.util.warningDisplay(
                "Cannot initialize without an original image sequence selected!"
            )
            return
        if not browserNode:
            browserNode = getFirstBrowser(origImageSequenceNode)
            if not browserNode:
                slicer.util.warningDisplay(
                    "Original image sequence is missing a browser node!"
                )

        # Remove any applied transform (because registration will reset this)
        origProxyNode = browserNode.GetProxyNode(origImageSequenceNode)
        self.removeParentTransform(origProxyNode)
        if not initialCarinaLandmark:
            initialCarinaLandmark = self.initializeCarinaLandmark()
        if not initialCropBox:
            browserNode.SetSelectedItemNumber(0)  # jump to first frame
            initialCropBox = self.initializeInitCropBox(refVolNode=origProxyNode)
        # For the output transform sequences, let's always create new ones when
        # initialization is triggered
        seqAndTypeAndName = (
            (nonLinAlignTformSeq1, "vtkMRMLTransformNode", "preTemplateNonLinTforms"),
            (nonLinAlignTformSeq2, "vtkMRMLTransformNode", "postTemplateNonLinTforms"),
            (
                carinaAlignmentTransformSeq,
                "vtkMRMLLinearTransformNode",
                "CarinaLinTforms",
            ),
        )
        for seq, seqType, name in seqAndTypeAndName:
            if seq:
                # rename old version before replacing
                seq.SetName(f"old_{seq.GetName()}")
            # Make new sequence node
            seq = scene.AddNewNodeByClass("vtkMRMLSequenceNode")
            setCompatibleSeqIndexing(seq, origImageSequenceNode)
            seq.SetAttribute("DataNodeClassName", seqType)
            uname = scene.GenerateUniqueName(name)
            seq.SetName(uname)

        # Set up organized outputs
        OutObj = namedtuple(
            "OutputData",
            (
                "origImageSequenceNode",
                "browserNode",
                "initialCarinaLandmark",
                "initialCropBox",
                "nonLinAlignTformSeq1",
                "nonLinAlignTformSeq2",
                "carinaAlignmentTransformSeq",
            ),
        )
        outputNamedTuple = OutObj(
            origImageSequenceNode=origImageSequenceNode,
            browserNode=browserNode,
            initialCarinaLandmark=initialCarinaLandmark,
            initialCropBox=initialCropBox,
            nonLinAlignTformSeq1=nonLinAlignTformSeq1,
            nonLinAlignTformSeq2=nonLinAlignTformSeq2,
            carinaAlignmentTransformSeq=carinaAlignmentTransformSeq,
        )
        return outputNamedTuple

    def removeParentTransform(self, node):
        """Remove any existing parent transform on the input node"""
        node.SetAndObserveTransformNodeID(None)

    def initializeInitCropBox(
        self,
        refVolNode: vtkMRMLScalarVolumeNode,
        initialCropBox: Optional[vtkMRMLMarkupsROINode] = None,
    ):
        """ """
        if not initialCropBox:
            scene = slicer.mrmlScene
            initCropBoxName = scene.GenerateUniqueName("InitialCropBox")
            initialCropBox = scene.AddNewNodeByClass(
                "vtkMRMLMarkupsROINode", initCropBoxName
            )
        # fit it to the volume
        roiMarkupFromVolumeNode(refVolNode, initialCropBox)
        # Shrink AP and LR extent by 1/2 as a starting box
        resize_roi_by_factors(initialCropBox, resize_factors=(0.5, 0.5, 1))
        # Adjust display
        dn = initialCropBox.GetDisplayNode()
        dn.SetSelectedColor(self.INIT_CROP_BOX_COLOR)
        dn.SetGlyphScale(1.5)
        dn.SetUseGlyphScale(True)
        dn.SetVisibility(True)
        return initialCropBox

    def initializeVolumeQuantBox(self, templateROINode, volumeQuantBox=None):
        """Create/standardize volume quantification ROI node"""
        if not volumeQuantBox:
            scene = slicer.mrmlScene
            volumeQuantBox = cloneItem(templateROINode)
            name = scene.GenerateUniqueName("VolumeQuantROI")
            volumeQuantBox.SetName(name)
            # Shrink a little to distinguish from template
            resize_roi_by_factors(volumeQuantBox, resize_factors=(0.95, 0.95, 0.85))
            # Automatically hide template roi from view?
            templateROINode.GetDisplayNode().SetVisibility(False)
        # Adjust display
        dn = volumeQuantBox.GetDisplayNode()
        dn.SetSelectedColor(self.VOLUME_QUANT_ROI_COLOR)
        dn.SetGlyphScale(1.5)
        dn.SetUseGlyphScale(True)
        dn.SetInteractionHandleScale(1.5)
        dn.SetTranslationHandleVisibility(True)
        dn.SetRotationHandleVisibility(False)
        dn.SetScaleHandleVisibility(True)
        dn.SetHandlesInteractive(True)
        dn.SetVisibility(True)
        return volumeQuantBox

    def initializeAlignedCropBox(self, templateROINode, alignedROINode=None):
        """Create/standardize aligned crop box ROI node"""
        if not alignedROINode:
            scene = slicer.mrmlScene
            alignedROINode = cloneItem(templateROINode)
            alignedCropBoxName = scene.GenerateUniqueName("AlignedCropBox")
            alignedROINode.SetName(alignedCropBoxName)
            # Shrink a little to distinguish from template
            resize_roi_by_factors(alignedROINode, resize_factors=(0.9, 0.9, 1.0))
            # Automatically hide template roi from view?
            templateROINode.GetDisplayNode().SetVisibility(False)
        # Adjust display
        dn = alignedROINode.GetDisplayNode()
        dn.SetSelectedColor(self.ALIGNED_CROP_BOX_COLOR)
        dn.SetGlyphScale(1.5)
        dn.SetUseGlyphScale(True)
        dn.SetVisibility(True)
        return alignedROINode

    def initializeInitMinIPSeg(
        self,
        minIPImage,
        initMinIPSeg=None,
    ):
        """Create and/or initialize the initial MinIP segmentation node.
        Create a minIPAirway segment by thresholding to -250 HU.
        Other facilitative steps could be added later if discovered...
        """
        scene = slicer.mrmlScene
        if not initMinIPSeg:
            initMinIPSeg = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            name = "InitialMinIPSeg"
            uname = scene.GenerateUniqueName(name)
            initMinIPSeg.SetName(uname)
        # Reset the segmentation node
        initMinIPSeg.GetSegmentation().RemoveAllSegments()
        # Create an initial airway segment by threshold
        segmentID = self.segmentByThreshold(
            initMinIPSeg,
            minIPImage,
            "initMinIP_Airway",
            thresholdMinValue=(-1100),
            thresholdMaxValue=(-250),
        )
        # Standardize segment color
        segment = initMinIPSeg.GetSegmentation().GetSegment(segmentID)
        segment.SetColor(self.INIT_MINIP_SEG_COLOR)
        # Create 3D surface
        initMinIPSeg.CreateClosedSurfaceRepresentation()
        return initMinIPSeg, segmentID

    def setCarinaLocationsLockedStatus(self, carinaLocSeq, lockedFlag: bool):
        """Lock/unlock being able to move the carina location
        fiducial points with the mouse.
        """
        browser = getFirstBrowser(carinaLocSeq)
        carinaNode = browser.GetProxyNode(carinaLocSeq)
        curFrameIdx = browser.GetSelectedItemNumber()
        for idx in range(carinaLocSeq.GetNumberOfDataNodes()):
            browser.SetSelectedItemNumber(idx)
            carinaNode.SetNthControlPointLocked(0, lockedFlag)
        browser.SetSelectedItemNumber(curFrameIdx)

    def setControlPointsLockedStatus(self, markupsNode, lockedFlag: bool):
        """Lock/unlock being able to move the markups control
        points with the mouse. Done on each individual control point.
        """
        for idx in range(markupsNode.GetNumberOfControlPoints()):
            markupsNode.SetNthControlPointLocked(idx, lockedFlag)

    def generateAlignedSegmentationSequence(
        self, imageSeg, minIPSeg, minIPSegmentID, outputSegSeq=None
    ) -> vtkMRMLSequenceNode:
        """Generate aligned segmentation sequence by thresholding each frame
        and intersecting it with the minIP airway segment."""
        outputSegSeq = self.generateSegmentationSequence(
            imageSeg,
            minIPSeg,
            minIPSegmentID,
            outputSegSeq,
            prefix="Aligned",
            segColor=self.ALIGNED_SEG_SEQ_COLOR,
        )
        return outputSegSeq

    def generateInitialSegmentationSequence(
        self, imageSeg, minIPSeg, minIPSegmentID, outputSegSeq=None
    ) -> vtkMRMLSequenceNode:
        """Generate initial segmentation sequence by thresholding each frame
        and intersecting it with the minIP airway segment."""
        outputSegSeq = self.generateSegmentationSequence(
            imageSeg,
            minIPSeg,
            minIPSegmentID,
            outputSegSeq,
            prefix="Initial",
            segColor=self.INIT_SEG_SEQ_COLOR,
        )
        return outputSegSeq

    def generateSegmentationSequence(
        self,
        imageSeq,
        minIPSeg,
        minIPSegmentID,
        outputSegSeq=None,
        prefix="AlignedOrInit",
        segColor=(1.0, 0, 0),
        largestIsland=True,
    ) -> vtkMRMLSequenceNode:
        """Generate segmentation sequence by thresholding each frame
        and intersecting it with the minIP airway segment."""
        scene = slicer.mrmlScene
        browser = getFirstBrowser(imageSeq)
        imageProxy = browser.GetProxyNode(imageSeq)
        if not outputSegSeq:
            outputSegSeq = scene.AddNewNodeByClass("vtkMRMLSequenceNode")
            name = f"{prefix}AirwaysSeq"
            uname = scene.GenerateUniqueName(name)
            outputSegSeq.SetName(uname)
            # I thought this was supposed to happen automatically, but seems like not...
            outputSegSeq.SetAttribute("DataNodeClassName", "vtkMRMLSegmentationNode")
        outputSegSeq.RemoveAllDataNodes()
        outputSegSeq.CopySequenceIndex(imageSeq)
        nFrames = imageSeq.GetNumberOfDataNodes()
        progressBar = createProgressDialog(
            value=0,
            maximum=nFrames,
            autoClose=True,
            windowTitle="Generating segmentation sequence...",
            labelText=f"Processing frame 0 of {nFrames}",
        )
        # Loop over frames
        for idx in range(nFrames):
            browser.SetSelectedItemNumber(idx)
            progressBar.value = idx
            progressBar.labelText = f"Processing frame {idx} of {nFrames-1}..."
            slicer.app.processEvents()
            frameSegNode = scene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            frameSegID = self.segmentByThreshold(
                frameSegNode,
                imageProxy,
                f"{prefix}Airway",
                thresholdMinValue=self.AIR_THRESH_MIN,
                thresholdMaxValue=self.AIR_THRESH_MAX,
            )
            self.limitSegmentRegion(frameSegID, frameSegNode, minIPSegmentID, minIPSeg)
            # Keep only largest island if requested
            if largestIsland:
                keep_largest_island(frameSegID, frameSegNode)
            # Assign segment color
            segment = frameSegNode.GetSegmentation().GetSegment(frameSegID)
            segment.SetColor(segColor)
            # Generate surface representation before storing/removal
            frameSegNode.CreateClosedSurfaceRepresentation()
            # Store result in sequence
            seqIdxValue = imageSeq.GetNthIndexValue(idx)
            outputSegSeq.SetDataNodeAtValue(frameSegNode, seqIdxValue)
            # Clean up
            scene.RemoveNode(frameSegNode)
        # Ensure sequence is added to browser node
        seqID = outputSegSeq.GetID()
        if not browser.IsSynchronizedSequenceNodeID(seqID):
            browser.AddSynchronizedSequenceNodeID(seqID)
            # Standardize settings and proxy node name
            browser.SetSaveChanges(outputSegSeq, True)
            browser.SetOverwriteProxyName(outputSegSeq, False)
            segProxy = browser.GetProxyNode(outputSegSeq)
            name = f"{prefix}AirwaySeg"
            uname = scene.GenerateUniqueName(name)
            segProxy.SetName(uname)
        # Close progress bar
        progressBar.value = nFrames
        return outputSegSeq

    def segmentMarginAdjust(self, segmentationNode, volNode, segmentID, marginMm=1):
        """Uses Margin tool to change the size of a segment by growing or shrinking
        on the outer margin. Negative values for marginMm will shrink the segment,
        positive values will grow it.  Note that the effect uses a kernel which
        must be a whole number of voxels in each dimension, so the actual dimension
        grown will be quantized and not exactly match the input value.
        """
        (
            segmentEditorWidget,
            segmentEditorNode,
            segmentationNode,
        ) = self.setup_segment_editor(segmentationNode)
        segmentEditorNode.SetSelectedSegmentID(segmentID)
        segmentEditorWidget.setActiveEffectByName("Margin")
        effect = segmentEditorWidget.activeEffect()
        # You can change parameters by calling: effect.setParameter("MyParameterName", someValue)
        effect.setParameter("MarginSizeMm", marginMm)
        # Masking settings (do not affect any other segments)
        segmentEditorNode.SetMaskMode(segmentationNode.EditAllowedEverywhere)
        segmentEditorNode.MasterVolumeIntensityMaskOff()
        segmentEditorNode.SetOverwriteMode(segmentEditorNode.OverwriteNone)
        # Apply the margin change
        effect.self().onApply()
        slicer.mrmlScene.RemoveNode(segmentEditorNode)

    def segmentByThreshold(
        self,
        segmentationNode,
        volumeNode,
        segmentName,
        thresholdMinValue,
        thresholdMaxValue,
    ):
        """Run segment editor threshold effect"""
        (
            segmentEditorWidget,
            segmentEditorNode,
            segmentationNode,
        ) = self.setup_segment_editor(segmentationNode, masterVolumeNode=volumeNode)
        # Find segmentID of segment or create if it doesn't exist
        segmentID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(
            segmentName
        )
        if not segmentID:
            # Create segment
            segmentID = segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
        segmentEditorNode.SetSelectedSegmentID(segmentID)
        # Select Threshold effect and parameters
        segmentEditorWidget.setActiveEffectByName("Threshold")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("MinimumThreshold", thresholdMinValue)
        effect.setParameter("MaximumThreshold", thresholdMaxValue)
        # Masking settings
        segmentEditorNode.SetMaskMode(segmentationNode.EditAllowedEverywhere)
        segmentEditorNode.SourceVolumeIntensityMaskOff()
        segmentEditorNode.SetOverwriteMode(segmentEditorNode.OverwriteNone)
        # Run thresholding
        effect.self().onApply()
        # Clean up
        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        return segmentID

    def setup_segment_editor(self, segmentationNode=None, masterVolumeNode=None):
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
            segmentEditorWidget.setSourceVolumeNode(masterVolumeNode)
        return segmentEditorWidget, segmentEditorNode, segmentationNode

    def findCarinas(
        self,
        airwaySegSeq: vtkMRMLSequenceNode,
        outputCarinaSeq: Optional[vtkMRMLSequenceNode] = None,
    ) -> "vtkMRMLSequenceNode":
        """Find carina locations based on a sequence of airway segments"""
        scene = slicer.mrmlScene
        browser = getFirstBrowser(airwaySegSeq)
        airwaySegNode = browser.GetProxyNode(airwaySegSeq)
        # Create output sequence if needed
        if not outputCarinaSeq:
            outputCarinaSeq = scene.AddNewNodeByClass("vtkMRMLSequenceNode")
            name = "CarinaLocationsSeq"
            uname = scene.GenerateUniqueName(name)
            outputCarinaSeq.SetName(uname)
            # I thought this was supposed to happen automatically, but seems like not...
            outputCarinaSeq.SetAttribute(
                "DataNodeClassName", "vtkMRMLMarkupsFiducialNode"
            )
        # Reset or set up the sequence
        outputCarinaSeq.RemoveAllDataNodes()
        outputCarinaSeq.CopySequenceIndex(airwaySegSeq)
        # Set up progress bar
        nFrames = browser.GetNumberOfItems()
        progressBar = createProgressDialog(
            value=0,
            maximum=nFrames,
            autoClose=True,
            windowTitle="Performing carina localization",
            labelText=f"Processing frame 0 of {nFrames}",
        )
        # Loop over frames
        for idx in range(nFrames):
            browser.SetSelectedItemNumber(idx)
            # Update progress bar
            progressBar.value = idx
            progressBar.labelText = f"Processing frame {idx} of {nFrames-1}"
            slicer.app.processEvents()
            # Always treat the first segment as the airway segment
            airwaySegID = airwaySegNode.GetSegmentation().GetNthSegmentID(0)
            airwayModel = convertSegmentToModelNode(airwaySegID, airwaySegNode)
            carinaLoc = self.findCarina(airwayModel)
            # Store in fiducial node
            tempFiducial = scene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", f"Carina_{idx}"
            )
            tempFiducial.SetControlPointLabelFormat(
                r"Carina"
            )  # no indexing, just force name
            tempFiducial.AddControlPointWorld(carinaLoc)
            # Store in sequence
            seqIndexValue = airwaySegSeq.GetNthIndexValue(idx)
            outputCarinaSeq.SetDataNodeAtValue(tempFiducial, seqIndexValue)
            # Clean up temporary nodes
            scene.RemoveNode(tempFiducial)
            scene.RemoveNode(airwayModel)
        # Add new sequence to browser
        seqID = outputCarinaSeq.GetID()
        if not browser.IsSynchronizedSequenceNodeID(seqID):
            browser.AddSynchronizedSequenceNodeID(seqID)
            # Standardize settings and proxy node name
            browser.SetSaveChanges(outputCarinaSeq, True)
            browser.SetOverwriteProxyName(outputCarinaSeq, False)
            segProxy = browser.GetProxyNode(outputCarinaSeq)
            name = f"Carina"
            uname = scene.GenerateUniqueName(name)
            segProxy.SetName(uname)
        # Close progress bar
        progressBar.value = nFrames

        return outputCarinaSeq

    def findCarina(self, airwayModel):
        """Given an airway model, find the carina point"""
        polydata = airwayModel.GetPolyData()
        smoothPolydata = smoothPolydataSurf(polydata)
        alignedPolyData, tForm = alignAirwayToTracheaCenterline(smoothPolydata)
        carina = findCarinaFromPolydata(alignedPolyData)
        tForm.Inverse()
        carina = tForm.TransformPoint(carina)
        return carina

    def runCarinaRegistration(self, inputs_here):
        """Run all steps of the multi-stage carina registration"""
        #### Pre-registration cropping with initial crop box
        #### Pre-Template Registrations
        #### Template construction
        #### Post-Template Registrations
        #### Construction of linear transforms
        #### Apply linear transform proxy to original image proxy

        raise NotImplementedError

    def createDraftTracheaCropBox(self, initialCropBox, resizeFactors=(0.8, 0.8, 1.0)):
        """Create draft tracheaCropBox ROI by copying the initialCropBox
        ROI and then shrinking it laterally a little bit (to distinguish)
        """
        logging.info("Creating draft tracheaCropBox...")
        tracheaCropBox = cloneItem(initialCropBox)
        scene = slicer.mrmlScene
        tracheaCropBox.SetName(scene.GenerateUniqueName("TracheaCropBox"))
        # Shrink it a little to distinguish from initial crop box
        resize_roi_by_factors(tracheaCropBox, resizeFactors)
        # Adjust display
        dn = tracheaCropBox.GetDisplayNode()
        dn.SetColor(self.TRACHEA_CROP_BOX_COLOR)
        dn.SetGlyphScale(1.0)
        dn.SetUseGlyphScale(True)
        # TODO: decide: Turn on rotation interaction handles here?
        # Also, hide the initial crop box to reduce confusion
        initialCropBox.GetDisplayNode().SetVisibility(False)
        # Log completion
        logging.info("Finished creating draft tracheaCropBox!")
        return tracheaCropBox

    def runCropImagesAndCreateMinIP(
        self,
        origImageSequenceNode=None,
        cropBox: vtkMRMLMarkupsROINode = None,
        transformSeq=None,
        outputCroppedSeq=None,
        outputMinIP=None,
        voxelSizeMm=0.5,
        showProgressBar=True,
    ):
        """This function handles creation of a cropped image sequence and
        the associated MinIP.
        """
        scene = slicer.mrmlScene
        if not origImageSequenceNode:
            raise ValueError("Missing origImageSequenceNode input!")
        browser = getFirstBrowser(origImageSequenceNode)
        if not cropBox:
            raise ValueError("Missing crop box ROI input!")
        ## Create output template image
        template = self.createVolumeFromROIandVoxelSize(
            cropBox, voxelSizeMm=voxelSizeMm
        )
        ## Initialize outputs as needed
        if not outputMinIP:
            outputMinIP = cloneItem(template)
            if not transformSeq:
                name = "InitialMinIP"
            else:
                name = "AlignedMinIP"
            uname = slicer.mrmlScene.GenerateUniqueName(name)
            outputMinIP.SetName(uname)
        # Initialize minIP with high values
        minIParr = arrayFromVolume(outputMinIP)
        minIParr[:] = 4000
        arrayFromVolumeModified(outputMinIP)
        # Initialize sequence for cropped images
        if not outputCroppedSeq:
            outputCroppedSeq = scene.AddNewNodeByClass("vtkMRMLSequenceNode")
            if not transformSeq:
                name = "InitialCroppedSeq"
            else:
                name = "AlignedCroppedSeq"
            uname = scene.GenerateUniqueName(name)
            outputCroppedSeq.SetName(uname)
            # I thought this was supposed to happen automatically, but seems like not...
            outputCroppedSeq.SetAttribute(
                "DataNodeClassName", "vtkMRMLScalarVolumeNode"
            )
        # Reset the output cropped sequence
        outputCroppedSeq.RemoveAllDataNodes()
        outputCroppedSeq.CopySequenceIndex(origImageSequenceNode)
        # Handle progress bar setup if requested
        nFrames = origImageSequenceNode.GetNumberOfDataNodes()
        if showProgressBar:
            progressBar = createProgressDialog(
                value=0,
                maximum=nFrames,
                autoClose=True,
                windowTitle="Cropping Images and Creating MinIP",
                labelText=f"Processing frame 0 of {nFrames-1}",
            )
        # Update the app process events, i.e. show the progress of the progress bar
        slicer.app.processEvents()
        # Loop over frames, accumulating minIP and making cropped image sequence
        # Adjust any display properties?
        for idx in range(nFrames):
            progressBar.value = idx
            progressBar.labelText = f"Processing frame {idx} of {nFrames-1}..."
            slicer.app.processEvents()
            browser.SetSelectedItemNumber(idx)
            origProxy = browser.GetProxyNode(origImageSequenceNode)
            tForm = browser.GetProxyNode(transformSeq) if transformSeq else None
            resampled = resampleVolumeUsingTemplate(
                origProxy, template, defaultValue=5000, applyTransform=tForm
            )
            # Store image
            seqIdxValue = origImageSequenceNode.GetNthIndexValue(idx)
            outputCroppedSeq.SetDataNodeAtValue(resampled, seqIdxValue)
            # Update minIP array
            arr = arrayFromVolume(resampled)
            lowerMask = arr < minIParr
            minIParr[lowerMask] = arr[lowerMask]
            # Clean up
            scene.RemoveNode(resampled)
        # Template no longer needed
        scene.RemoveNode(template)
        # Update minIP node
        arrayFromVolumeModified(outputMinIP)
        minVal = np.min(minIParr[:])
        maxVal = np.max(minIParr[:])
        outputMinIP.GetDisplayNode().SetWindowLevelMinMax(minVal, maxVal)
        # Close progress bar
        progressBar.value = nFrames
        # Add new sequence to browser (unless already there)
        seqID = outputCroppedSeq.GetID()
        if not browser.IsSynchronizedSequenceNodeID(seqID):
            browser.AddSynchronizedSequenceNodeID(seqID)
            # Standardize settings and proxy node name
            browser.SetSaveChanges(outputCroppedSeq, True)
            browser.SetOverwriteProxyName(outputCroppedSeq, False)
            segProxy = browser.GetProxyNode(outputCroppedSeq)
            if not transformSeq:
                name = "InitCroppedImage"
            else:
                name = "AlignedCroppedImage"
            uname = scene.GenerateUniqueName(name)
            segProxy.SetName(uname)
        # Organize outputs
        outputs = {"croppedSeq": outputCroppedSeq, "minIP": outputMinIP}
        return outputs

    def makeMinIPFromSequence(self, seqNode, outputMinIP=None, useTranforms=True):
        """Calculate a temporal minIP of an image sequence. If useTransforms is
        True, then any parent transforms on the image sequence are applied before
        sampling output voxel locations. If useTransforms is False, then any
        parent transforms are ignored.

        If an outputMinIP volume is supplied, it is used as the resampling geometry
        reference.  If not supplied, it is created with geometry to match the
        (possibly transformed) current frame of the image sequence.
        """
        browser = getFirstBrowser(seqNode)
        proxNode = browser.GetProxyNode(seqNode)
        scene = slicer.mrmlScene

        if not outputMinIP:
            outputMinIP = cloneItem(proxNode)
            uname = scene.GenerateUniqueName("MinIP")
            outputMinIP.SetName(uname)
            if useTranforms:
                outputMinIP.SetAndObserveTransformNodeID(proxNode.GetTransformID())
                outputMinIP.HardenTransform()

        # Initialize the minIP with high values
        minIParr = arrayFromVolume(outputMinIP)
        minIParr[:] = 4000

        # Loop over sequence frames and accumulate minimum values
        for frameIdx in range(seqNode.GetNumberOfDataNodes):
            browser.SetSelectedItemNumber(frameIdx)
            if useTranforms:
                tForm = slicer.util.GetNodeByID(proxNode.GetTransformNodeID())
            else:
                tForm = None
            # Resample into minIP geometry; use high values for outside image
            resampled = resampleVolumeUsingTemplate(
                proxNode,
                templateVolNode=outputMinIP,
                defaultValue=5000,
                applyTransform=tForm,
            )
            resampled_arr = arrayFromVolume(resampled)
            mask = resampled_arr < minIParr
            minIParr[mask] = resampled_arr[mask]
            # Clean up resampled
            scene.RemoveNode(resampled)
        # Update the minIP
        arrayFromVolumeModified(outputMinIP)

        return outputMinIP

    def runExtractCenterline(
        self, inputSurface, inputSegmentID, endPoints, rawCenterline=None
    ):
        """Trying to mimic the implementation in ExtractCenterline.py from VMTK module"""
        import ExtractCenterline

        extractLogic = ExtractCenterline.ExtractCenterlineLogic()
        ## Preprocess ##
        inputSurfacePolyData = extractLogic.polyDataFromNode(
            inputSurface, inputSegmentID
        )
        if not inputSurfacePolyData or inputSurfacePolyData.GetNumberOfPoints() == 0:
            raise ValueError("Valid input surface is required")
        # These are the default values in the VMTK module, which I have never had to change (until I ran into an error)
        # OK, now I've had to change decimationAggressiveness down to 3.5 from 4.0.  Preprocessed surface had inverted segments after decimation
        # which lead to failed centerline curve.  It is possible that we could detect failed curve if it has only two control points? (That was the
        # case in the single failure so far). If so, we could re-run with lower decimation aggressiveness.
        preprocessEnabled = True  # (self._parameterNode.GetParameter("PreprocessInputSurface") == "true")
        targetNumberOfPoints = (
            5000.0  # float(self._parameterNode.GetParameter("TargetNumberOfPoints"))
        )
        decimationAggressiveness = 3.5  # 4.0 #float(self._parameterNode.GetParameter("DecimationAggressiveness"))
        subdivideInputSurface = False  # (self._parameterNode.GetParameter("SubdivideInputSurface") == "true")
        slicer.util.showStatusMessage(
            "Preprocessing surface before centerline extraction..."
        )
        slicer.app.processEvents()
        preprocessedPolyData = extractLogic.preprocess(
            inputSurfacePolyData,
            targetNumberOfPoints,
            decimationAggressiveness,
            subdivideInputSurface,
        )
        ##
        scene = slicer.mrmlScene
        if not rawCenterline:
            rawCenterline = scene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode")
            uname = scene.GenerateUniqueName("RawCenterline")
            rawCenterline.SetName(uname)
        # Standardize display propertiess
        dn = rawCenterline.GetDisplayNode()
        dn.SetColor(self.RAW_CENTERLINE_COLOR)
        dn.SetGlyphScale(1.0)
        dn.SetUseGlyphScale(True)

        slicer.util.showStatusMessage("Extracting centerline...")
        slicer.app.processEvents()  # force update
        centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(
            preprocessedPolyData, endPoints
        )
        centerlinePropertiesTableNode = None
        extractLogic.createCurveTreeFromCenterline(
            centerlinePolyData, rawCenterline, centerlinePropertiesTableNode
        )
        if rawCenterline.GetNumberOfControlPoints() == 2:
            # Extraction had an error, likely due to too high decimation aggressiveness
            # Try again?
            slicer.util.errorDisplay(
                "Centerline generation failed, possibly due to decimationAggressiveness being too high."
            )
            # TODO TODO TODO: ?implement automatically trying again, with message to user about what's going on
            return None
        return rawCenterline

    def smoothCenterline(self, rawCenterline, smoothCenterline=None):
        """ """
        smoothCenterline = smoothCurveNode(
            rawCenterline, outputSmoothCurveNode=smoothCenterline
        )
        uname = slicer.mrmlScene.GenerateUniqueName("SmoothedCenterline")
        smoothCenterline.SetName(uname)
        dn = smoothCenterline.GetDisplayNode()
        dn.SetSelectedColor(self.SMOOTH_CENTERLINE_COLOR)
        dn.SetGlyphScale(1.0)
        dn.SetUseGlyphScale(True)
        return smoothCenterline

    def resampleCurveNode(
        self,
        curvePointsNode: vtkMRMLMarkupsCurveNode,
        pathStepSpacingMm: float = 0.5,
        resampledOutputCurveNode: Optional[vtkMRMLMarkupsCurveNode] = None,
    ) -> vtkMRMLMarkupsCurveNode:
        """
        Uniformly resamples the input curve's control points
        """
        vtkResampledPoints = vtk.vtkPoints()  # initialize
        vtkMRMLMarkupsCurveNode.ResamplePoints(
            curvePointsNode.GetCurvePointsWorld(),
            vtkResampledPoints,
            pathStepSpacingMm,
            curvePointsNode.GetCurveClosed(),
        )
        resampledControlPoints = vtk.util.numpy_support.vtk_to_numpy(
            vtkResampledPoints.GetData()
        )
        # Create output curve node if not supplied
        if resampledOutputCurveNode is None:
            resampledOutputCurveNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsCurveNode",
                slicer.mrmlScene.GenerateUniqueName(
                    f"{curvePointsNode.GetName()}_resampled"
                ),
            )
        # Update output with resampled curve points
        slicer.util.updateMarkupsControlPointsFromArray(
            resampledOutputCurveNode, resampledControlPoints
        )
        return resampledOutputCurveNode

    def standardizeCenterlineOrientation(self, centerline, carinaMarkup):
        """Standardizes the order of centerline control points by reversing
        if the carina is closer to the beginning than the end.
        """
        cp_arr = arrayFromMarkupsControlPoints(centerline, world=True)
        cp0 = cp_arr[0]
        cpLast = cp_arr[-1]
        carinaLoc = arrayFromMarkupsControlPoints(carinaMarkup, world=True)[0]
        dist0 = np.linalg.norm(carinaLoc - cp0)
        distL = np.linalg.norm(carinaLoc - cpLast)
        if dist0 < distL:
            # Beginning of centerline curve is closer to the carina than
            # the end of the curve, we need to reverse the control point
            # order
            slicer.util.updateMarkupsControlPointsFromArray(
                centerline, np.flipud(cp_arr), world=True
            )

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
        allCSAs = self.convertTableNodeToNumpy(mergedTableNode, skipRASCols=True)
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
        vInvSCoordCol = vtk.vtkFloatArray()
        vInvSCoordCol.SetName("InvS_Coordinate")
        vZInvSCoordCol = vtk.vtkFloatArray()
        vZInvSCoordCol.SetName("ZInvS_Coordinate")
        maxS = np.max(rasCoords[:, 2])
        for idx in range(lowestCSA.size):
            vLowestCol.InsertNextValue(lowestCSA[idx])
            vHighestCol.InsertNextValue(highestCSA[idx])
            S = rasCoords[idx, 2]
            invS = -S
            zInvS = maxS - S
            vSupCoordCol.InsertNextValue(S)
            vInvSCoordCol.InsertNextValue(invS)
            vZInvSCoordCol.InsertNextValue(zInvS)
        # Create table node to hold envelope
        envTableNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTableNode", slicer.mrmlScene.GenerateUniqueName("CSAEnvelopeTable")
        )
        envTableNode.AddColumn(vLowestCol)
        envTableNode.AddColumn(vHighestCol)
        envTableNode.AddColumn(vSupCoordCol)
        envTableNode.AddColumn(vInvSCoordCol)
        envTableNode.AddColumn(vZInvSCoordCol)
        # Make plot series for each of these
        lowPlotSeries = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "MinCSA"
        )
        lowPlotSeries.SetAndObserveTableNodeID(envTableNode.GetID())
        lowPlotSeries.SetXColumnName("InvS_Coordinate")
        lowPlotSeries.SetYColumnName("MinCSA")
        lowPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        gray = [0.75, 0.75, 0.75]
        lowPlotSeries.SetColor(gray)
        highPlotSeries = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "MaxCSA"
        )
        highPlotSeries.SetAndObserveTableNodeID(envTableNode.GetID())
        highPlotSeries.SetXColumnName("InvS_Coordinate")
        highPlotSeries.SetYColumnName("MaxCSA")
        highPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        highPlotSeries.SetColor(gray)
        # Make plot series for dynamic one using the proxy node
        dynPlotSeries = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", "FrameCSA"
        )
        dynPlotSeries.SetAndObserveTableNodeID(proxyTableNode.GetID())
        dynPlotSeries.SetXColumnName("InvS_coord")
        dynPlotSeries.SetYColumnName("Cross-section area")
        dynPlotSeries.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        dynPlotSeries.SetColor([0, 0, 1])
        # Set up plotChart
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        plotChartNode.SetName(slicer.mrmlScene.GenerateUniqueName("CSA_Dynamic"))
        for plotSeries in [lowPlotSeries, highPlotSeries, dynPlotSeries]:
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeries.GetID())
        plotChartNode.SetXAxisTitle("-S-Coordinate (Superior --> Inferior, mm)")
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
            elif skipRASCols and vtkArr.GetName() in [
                "R_coord",
                "A_coord",
                "S_coord",
                "InvS_coord",
                "ZInvS_coord",
            ]:
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
        # Add inverse S coord col and zeroed inverse S column to table
        invSCol = vtk.vtkFloatArray()
        invSCol.SetName("InvS_coord")
        zInvSCol = vtk.vtkFloatArray()
        zInvSCol.SetName("ZInvS_coord")
        sValues = numpy_support.vtk_to_numpy(sCol)
        maxS = np.max(sValues)
        for S in sValues:
            invSCol.InsertNextValue(-S)
            zInvSCol.InsertNextValue(maxS - S)

        # Also add these columns to the tables in the sequence
        # (so that they can be plotted against)
        for frameIdx in range(tableSequence.GetNumberOfDataNodes()):
            tableNode = tableSequence.GetNthDataNode(frameIdx)
            tableNode.AddColumn(invSCol)
            tableNode.AddColumn(zInvSCol)

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
        """Compute segment volumes.  Currently just finds volume_mm3"""
        import SegmentStatistics

        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        spn = segStatLogic.getParameterNode()
        spn.SetParameter("Segmentation", segmentationNode.GetID())
        spn.SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", "False")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.enabled", "True")
        spn.SetParameter("ClosedSurfaceSegmentStatisticsPlugin.enabled", "False")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.voxel_count.enabled", "True")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.volume_mm3.enabled", "True")
        spn.SetParameter("LabelmapSegmentStatisticsPlugin.volume_cm3.enabled", "False")
        # Actually do the computation
        segStatLogic.computeStatistics()
        return segStatLogic

    def setCompatibleSeqIndexing(self, newSeqNode, seqNodeToMatch):
        """Sets indexing to be compatible by matching the index
        name, type, and units to match an existing sequence node.
        This is necessary for them to share a single sequence browser
        and be synchronized.
        ( NO LONGER NEEDED, vtkMRMLSequenceNode.SequenceIndex()
        now does exactly this)
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

    def cropAndResampleSequence(
        self,
        inputVolumeSeq,
        roiNode,
        outputVolumeSeq=None,
        voxelSizeMm=[0.5],
        defaultVoxelValue=0,
    ):
        """Use an roi node and desired voxel resolution to create a template volume, and then
        resample each frame of the inputVolumeSeq to that template geometry.
        """
        templateVol = self.createVolumeFromROIandVoxelSize(
            roiNode, voxelSizeMm=voxelSizeMm
        )
        browserNode = (
            slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(
                inputVolumeSeq
            )
        )
        if outputVolumeSeq is None:
            outputVolumeSeq = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSequenceNode", "Cropped"
            )
            outputVolumeSeq.CopySequenceIndex(inputVolumeSeq)
        # Loop over sequence frames
        for idx in range(inputVolumeSeq.GetNumberOfDataNodes()):
            inputVol = inputVolumeSeq.GetNthDataNode(idx)
            resampledVol = resampleVolumeUsingTemplate(
                inputVol, templateVol, defaultValue=defaultVoxelValue
            )
            inputIdxValue = inputVolumeSeq.GetNthIndexValue(idx)
            outputVolumeSeq.SetDataNodeAtValue(resampledVol, inputIdxValue)
            # Clean up resampled volume (outside of the sequence)
            slicer.mrmlScene.RemoveNode(resampledVol)
        # Add the new sequence to the browser
        browserNode.AddSynchronizedSequenceNodeID(outputVolumeSeq.GetID())
        browserNode.SetSaveChanges(outputVolumeSeq, True)
        # Clean up template volume node
        slicer.mrmlScene.RemoveNode(templateVol)
        return outputVolumeSeq

    def createVolumeFromROIandVoxelSize(
        self, ROINode, voxelSizeMm=[1.0, 1.0, 1.0], prioritizeVoxelSize=True
    ):
        """Create an empty scalar volume node with the given resolution, location, and
        orientation. The resolution must be given directly (single or scalar value interpreted
        as an isotropic edge length), and the location, size, and orientation are derived from
        the ROINode (a vtkMRMLAnnotationROINode). If prioritizeVoxelSize is True (the default),
        and the size of the ROI is not already an integer number of voxels across in each dimension,
        the ROI is minimally expanded to the next integer number of voxels across in each dimension.
        If prioritizeVoxelSize is False, then the ROI is left unchanged, and the voxel dimensions
        are minimally adjusted such that the existing ROI is an integer number of voxels across.
        """
        import numpy as np

        # Ensure resolutionMm can be converted to a list of 3 voxel edge lengths
        # If voxel size is a scalar or a one-element list, interpret that as a request for
        # isotropic voxels with that edge length
        if hasattr(voxelSizeMm, "__len__"):
            if len(voxelSizeMm) == 1:
                voxelSizeMm = [voxelSizeMm[0]] * 3
            elif not len(voxelSizeMm) == 3:
                raise Exception(
                    "voxelSizeMm must either have one or 3 elements; it does not."
                )
        else:
            try:
                v = float(voxelSizeMm)
                voxelSizeMm = [v] * 3
            except:
                raise Exception(
                    "voxelSizeMm does not appear to be a number or a list of one or three numbers."
                )

        # Resolve any tension between the ROI size and resolution if ROI is not an integer number of voxels in all dimensions
        ROIRadiusXYZMm = [0] * 3  # initialize
        ROINode.GetRadiusXYZ(ROIRadiusXYZMm)  # fill in ROI sizes
        ROIDiamXYZMm = 2 * np.array(
            ROIRadiusXYZMm
        )  # need to double radii to get box dims
        numVoxelsAcrossFloat = np.divide(ROIDiamXYZMm, voxelSizeMm)
        voxelTol = 0.1  # fraction of a voxel it is OK to shrink the ROI by (rather than growing by 1-voxelTol voxels)
        if prioritizeVoxelSize:
            # Adjust ROI size by increasing it to the next integer multiple of the voxel edge length
            numVoxAcrossInt = []
            for voxAcross in numVoxelsAcrossFloat:
                # If over by less voxelTol of a voxel, don't ceiling it
                diff = voxAcross - np.round(voxAcross)
                if diff > 0 and diff < voxelTol:
                    voxAcrossInt = np.round(
                        voxAcross
                    )  # round it down, which will shrink the ROI by up to voxelTol voxels
                else:
                    voxAcrossInt = np.ceil(
                        voxAcross
                    )  # otherwise, grow ROI to the next integer voxel size
                numVoxAcrossInt.append(voxAcrossInt)
            # Figure out new ROI dimensions
            adjustedROIDiamXYZMm = np.multiply(numVoxAcrossInt, voxelSizeMm)
            adjustedROIRadiusXYZMm = (
                0.5 * adjustedROIDiamXYZMm
            )  # radii are half box dims
            # Apply adjustment
            ROINode.SetRadiusXYZ(adjustedROIRadiusXYZMm)
        else:  # prioritize ROI dimension, adjust voxel resolution
            numVoxAcrossInt = np.round(numVoxelsAcrossFloat)
            # Adjust voxel resolution
            adjustedVoxelSizeMm = np.divide(ROIDiamXYZMm, numVoxAcrossInt)
            voxelSizeMm = adjustedVoxelSizeMm

        #
        volumeName = "OutputTemplateVolume"
        voxelType = (
            vtk.VTK_INT
        )  # not sure if this locks in anything for resampling, if so, might be an issue
        imageDirections, origin = self.getROIDirectionsAndOrigin(
            ROINode
        )  # these are currently not normalized!

        # Create volume node
        templateVolNode = self.createVolumeNodeFromScratch(
            volumeName,
            imageSizeVox=numVoxAcrossInt,
            imageOrigin=origin,
            imageSpacingMm=voxelSizeMm,
            imageDirections=imageDirections,
            voxelType=voxelType,
        )
        return templateVolNode

    def getROIDirectionsAndOrigin(self, roiNode):
        import numpy as np

        # Processing is different depending on whether the roiNode is AnnotationsMarkup or MarkupsROINode
        if isinstance(roiNode, slicer.vtkMRMLMarkupsROINode):
            axis0 = [0, 0, 0]
            roiNode.GetXAxisWorld(
                axis0
            )  # This respects soft transforms applied to the ROI!
            axis1 = [0, 0, 0]
            roiNode.GetYAxisWorld(axis1)
            axis2 = [0, 0, 0]
            roiNode.GetZAxisWorld(axis2)
            # These axes are the columns of the IJKToRAS directions matrix, but when
            # we supply a list of directions to the imageDirections, it takes a list of rows,
            # so we need to transpose
            directions = np.transpose(
                np.stack((axis0, axis1, axis2))
            )  # for imageDirections
            center = [0, 0, 0]
            roiNode.GetCenterWorld(center)
            radiusXYZ = [0, 0, 0]
            roiNode.GetRadiusXYZ(radiusXYZ)
            # The origin in the corner where the axes all point along the ROI
            origin = (
                np.array(center)
                - np.array(axis0) * radiusXYZ[0]
                - np.array(axis1) * radiusXYZ[1]
                - np.array(axis2) * radiusXYZ[2]
            )
        else:
            # Input is not markupsROINode, must be older annotations ROI instead
            T_id = roiNode.GetTransformNodeID()
            if T_id:
                T = slicer.mrmlScene.GetNodeByID(T_id)
            else:
                T = None
            if T:
                # Transform node is present
                # transformMatrix = slicer.util.arrayFromTransformMatrix(T) # numpy 4x4 array
                # if nested transform, then above will fail! # TODO TODO
                worldToROITransformMatrix = vtk.vtkMatrix4x4()
                T.GetMatrixTransformBetweenNodes(None, T, worldToROITransformMatrix)
                # then convert to numpy
            else:
                worldToROITransformMatrix = (
                    vtk.vtkMatrix4x4()
                )  # defaults to identity matrix
                # transformMatrix = np.eye(4)
            # Convert to directions (for image directions)
            axis0 = np.array(
                [worldToROITransformMatrix.GetElement(i, 0) for i in range(3)]
            )
            axis1 = np.array(
                [worldToROITransformMatrix.GetElement(i, 1) for i in range(3)]
            )
            axis2 = np.array(
                [worldToROITransformMatrix.GetElement(i, 2) for i in range(3)]
            )
            directions = (axis0, axis1, axis2)  # for imageDirections
            # Find origin of roiNode (RAS world coord)
            # Origin is Center - radius1 * direction1 - radius2 * direction2 - radius3 * direction3
            ROIToWorldTransformMatrix = vtk.vtkMatrix4x4()
            ROIToWorldTransformMatrix.DeepCopy(worldToROITransformMatrix)  # copy
            ROIToWorldTransformMatrix.Invert()  # invert worldToROI to get ROIToWorld
            # To adjust the origin location I need to use the axes of the ROIToWorldTransformMatrix
            ax0 = np.array(
                [ROIToWorldTransformMatrix.GetElement(i, 0) for i in range(3)]
            )
            ax1 = np.array(
                [ROIToWorldTransformMatrix.GetElement(i, 1) for i in range(3)]
            )
            ax2 = np.array(
                [ROIToWorldTransformMatrix.GetElement(i, 2) for i in range(3)]
            )
            boxDirections = (ax0, ax1, ax2)
            TransformOrigin4 = ROIToWorldTransformMatrix.MultiplyPoint([0, 0, 0, 1])
            TransformOrigin = TransformOrigin4[:3]
            roiCenter = [0] * 3  # intialize
            roiNode.GetXYZ(roiCenter)  # fill
            # I want to transform the roiCenter using roiToWorld
            transfRoiCenter4 = ROIToWorldTransformMatrix.MultiplyPoint([*roiCenter, 1])
            transfRoiCenter = transfRoiCenter4[:3]
            # Now need to subtract
            radXYZ = [0] * 3
            roiNode.GetRadiusXYZ(radXYZ)
            origin = (
                np.array(transfRoiCenter)
                - ax0 * radXYZ[0]
                - ax1 * radXYZ[1]
                - ax2 * radXYZ[2]
            )

        # Return outputs
        return directions, origin

    def createVolumeNodeFromScratch(
        self,
        nodeName="VolumeFromScratch",
        imageSizeVox=(256, 256, 256),  # image size in voxels
        imageSpacingMm=(2.0, 2.0, 2.0),  # voxel size in mm
        imageOrigin=(0.0, 0.0, 0.0),
        imageDirections=(
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ),  # Image axis directions IJK to RAS,  (these should be orthogonal!)
        fillVoxelValue=0,
        voxelType=vtk.VTK_INT,
    ):
        """Create a scalar volume node from scratch, given information on
        name, size, spacing, origin, image directions, fill value, and voxel
        type.
        """
        imageData = vtk.vtkImageData()
        imageSizeVoxInt = [int(v) for v in imageSizeVox]
        imageData.SetDimensions(imageSizeVoxInt)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Normalize and check orthogonality image directions
        import numpy as np
        import logging

        imageDirectionsUnit = [np.divide(d, np.linalg.norm(d)) for d in imageDirections]
        angleTolDegrees = 1  # allow non-orthogonality up to 1 degree
        for pair in ([0, 1], [1, 2], [2, 0]):
            angleBetween = np.degrees(
                np.arccos(
                    np.dot(imageDirectionsUnit[pair[0]], imageDirectionsUnit[pair[1]])
                )
            )
            if abs(90 - angleBetween) > angleTolDegrees:
                logging.warning(
                    "Warning! imageDirections #%i and #%i supplied to createVolumeNodeFromScratch are not orthogonal!"
                    % (pair[0], pair[1])
                )
                # Continue anyway, because volume nodes can sort of handle non-orthogonal image directions (though they're not generally expected)
        # Create volume node
        volumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", nodeName
        )
        volumeNode.SetOrigin(imageOrigin)
        volumeNode.SetSpacing(imageSpacingMm)
        volumeNode.SetIJKToRASDirections(imageDirections)
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()
        return volumeNode

    def croppedSequenceFromPoints(
        self,
        markupsPoints,
        inputVolumeSequence,
        sliceSizeMm=20,
        longAxisExtensionsMm=(0, 20),
        roiNode=None,
        voxelSizeMm=0.5,
        outputCroppedVolumeSequence=None,
        defautVoxelValue=0,
    ):
        """Make a cropped volume sequence using an ROI derived from the first and last
        control points of an input markups node. Lots of options which are passed along
        to subfunctions.
        """
        roiNode = makeROIFromPoints(
            markupsPoints,
            sliceSizeMm=sliceSizeMm,
            longAxisExtensionsMm=longAxisExtensionsMm,
            outputROINode=roiNode,
        )
        outputCroppedVolumeSequence = self.cropAndResampleSequence(
            inputVolumeSequence,
            roiNode,
            outputCroppedVolumeSequence,
            voxelSizeMm=voxelSizeMm,
            defaultVoxelValue=defautVoxelValue,
        )
        return outputCroppedVolumeSequence, roiNode


#
# Helper functions
#


def makeROIFromPoints(
    markupsNode, sliceSizeMm=20, longAxisExtensionsMm=(0, 20), outputROINode=None
):
    """Create or update a markupsROI node based on the first and last control points
    of an input markups node.  The ROI has one axis aligned along the line connecting those
    two points, and the other two axes perpedicular to it.  The transverse dimensions
    are set by sliceSizeMm, which defaults to 20mm. The long axis is extended outward
    from the given points by lengths given by longAxisExtensions. The other ROI axis
    directions are given by the projection of the R and A vectors into the plane
    perpendicular to the axis. There is an implicit assumption that the long axis
    runs more superior-inferior than other directions, but may work fine more generally.
    """
    if outputROINode is None:
        outputROINode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsROINode", "CropROI"
        )
    # Get long axis-defining points
    cpArr = slicer.util.arrayFromMarkupsControlPoints(markupsNode)
    cpStart = cpArr[0, :]
    cpEnd = cpArr[-1, :]
    # Extend axis
    v = cpEnd - cpStart
    u = v / np.linalg.norm(v)
    axisStart = cpStart - u * longAxisExtensionsMm[0]
    axisEnd = cpEnd + u * longAxisExtensionsMm[1]
    # Find ROI Center
    roiCenter = (axisStart + axisEnd) / 2
    # Find ROI size
    axisLength = np.linalg.norm(axisEnd - axisStart)
    roiSize = (sliceSizeMm, sliceSizeMm, axisLength)
    # Construct the ObjectToNode matrix
    # Z axis should be S axis projected onto long axis
    sVect = np.array([0, 0, 1])
    zAxis = projectVector(sVect, v)
    zAxis = zAxis / np.linalg.norm(zAxis)
    # X axis should be R vector projected to perpendicular plane
    rVect = np.array([1, 0, 0])
    xAxis = rVect - projectVector(rVect, zAxis)
    xAxis = xAxis / np.linalg.norm(xAxis)
    # Y axis should be Long axis cross X vector
    yAxis = np.cross(zAxis, xAxis)
    yAxis = yAxis / np.linalg.norm(yAxis)
    # Assemble into ObjectToNodeMatrix
    objToNodeMatrix = np.eye(4)
    objToNodeMatrix[0:3, 0] = xAxis
    objToNodeMatrix[0:3, 1] = yAxis
    objToNodeMatrix[0:3, 2] = zAxis
    objToNodeMatrix[0:3, 3] = roiCenter
    vObjToNodeMatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vObjToNodeMatrix.SetElement(i, j, objToNodeMatrix[i, j])
    # Set output roi node properties
    outputROINode.SetAndObserveObjectToNodeMatrix(vObjToNodeMatrix)
    outputROINode.SetSize(*roiSize)

    return outputROINode


def projectVector(a, b):
    """project the first vector onto the second"""
    projAontoB = (np.dot(a, b) / np.dot(b, b)) * b
    return projAontoB


def resampleVolumeUsingTemplate(
    inputVolNode,
    templateVolNode,
    outputResampledVolNode=None,
    defaultValue=0,
    applyTransform=None,
):
    """Resample an image volume using another volume as a template for the geometry.
    Any voxels in the template but outside the input are assigned the default value.
    If a transform is included (i.e. if applyTransform is a transform node and not
    None), then it is applied to the input volume before resampling occurs.
    Uses the brainsresample module internally to carry out the transforming and
    resampling.  In this DynamicMalaciaTools module, this function will be used
    primarily as a method of cropping while resampling.
    """
    if outputResampledVolNode is None:
        outName = inputVolNode.GetName() + "_templateResampled"
        outputResampledVolNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode", outName
        )
    params = {
        "inputVolume": inputVolNode.GetID(),
        "referenceVolume": templateVolNode.GetID(),
        "outputVolume": outputResampledVolNode.GetID(),
        "interpolationMode": "Linear",
        "pixelType": "input",
        "defaultValue": defaultValue,
        "warpTransform": applyTransform.GetID() if applyTransform is not None else "",
    }
    cliNode = slicer.cli.runSync(slicer.modules.brainsresample, None, params)
    # Return the resampled volume
    return outputResampledVolNode


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


def roiMarkupFromVolumeNode(volumeNode, outputRoiNode=None):
    """Create or update markupsROI node and fit to volume node"""
    if outputRoiNode is None:
        outputRoiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")

    cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLCropVolumeParametersNode"
    )
    cropVolumeParameters.SetInputVolumeNodeID(volumeNode.GetID())
    cropVolumeParameters.SetROINodeID(outputRoiNode.GetID())
    slicer.modules.cropvolume.logic().SnapROIToVoxelGrid(
        cropVolumeParameters
    )  # optional (rotates the ROI to match the volume axis directions)
    slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)
    slicer.mrmlScene.RemoveNode(cropVolumeParameters)
    return outputRoiNode


def resize_roi_by_factors(roi_node, resize_factors):
    """
    Resizes a markupsROINode by the given resizing factors along the left-right,
    anterior-posterior, and superior-inferior axes.

    Args:
        roi_node: The markupsROINode to resize.
        resize_factors: A tuple of three floats representing the resizing factors
                        for the left-right, anterior-posterior, and superior-inferior axes,
                        respectively.

    NOTE: this function considers the final effects on orientation of any
    applied parent transforms before choosing which axis is best aligned.
    However, resizing is applied to the base ROI object itself, so if a
    nonlinear parent transform is applied, the transformed size may not
    change by exactly the supplied factors.
    """

    if not isinstance(roi_node, slicer.vtkMRMLMarkupsROINode):
        raise ValueError("Input node must be a markupsROINode.")

    if not isinstance(resize_factors, tuple) or len(resize_factors) != 3:
        raise ValueError("Resize factors must be a tuple of three floats.")

    for factor in resize_factors:
        if not isinstance(factor, (int, float)):
            raise ValueError("Resize factors must be numeric.")

    # Get the current ROI dimensions and center
    dimensions = roi_node.GetSize()
    center = roi_node.GetCenter()

    # Get the object to world matrix
    # object_to_world = vtk.vtkMatrix4x4()
    object_to_world = roi_node.GetObjectToWorldMatrix()

    # Get the axis vectors in world coordinates
    axis_vectors = (
        (object_to_world.GetElement(r, 0) for r in range(3)),
        (object_to_world.GetElement(r, 1) for r in range(3)),
        (object_to_world.GetElement(r, 2) for r in range(3)),
    )

    # Define the reference axes in world coordinates
    left_right_axis = [1, 0, 0]
    anterior_posterior_axis = [0, 1, 0]
    superior_inferior_axis = [0, 0, 1]

    # Calculate the dot products to determine the alignment of ROI axes with reference axes
    dot_products = []
    for axis_vector in axis_vectors:
        dot_products.append(
            [
                abs(sum(a * b for a, b in zip(axis_vector, left_right_axis))),
                abs(sum(a * b for a, b in zip(axis_vector, anterior_posterior_axis))),
                abs(sum(a * b for a, b in zip(axis_vector, superior_inferior_axis))),
            ]
        )

    # Assign resizing factors to ROI axes based on alignment
    axis_indices = list(range(3))
    resized_dimensions = list(dimensions)
    assigned_factors = [
        False,
        False,
        False,
    ]  # Track if a factor has been assigned to prevent duplicate assignment.

    for factor_index, factor in enumerate(resize_factors):
        max_dot_product = -1
        best_axis_index = -1
        for roi_axis_index in axis_indices:
            if assigned_factors[factor_index]:
                break
            if dot_products[roi_axis_index][factor_index] > max_dot_product:
                max_dot_product = dot_products[roi_axis_index][factor_index]
                best_axis_index = roi_axis_index

        if best_axis_index != -1:
            resized_dimensions[best_axis_index] = dimensions[best_axis_index] * factor
            axis_indices.remove(best_axis_index)
            assigned_factors[factor_index] = True

    # Set the new ROI dimensions
    roi_node.SetSize(resized_dimensions)
    roi_node.SetCenter(center)  # Center remains the same
    return  # no return, input is modified


def setCompatibleSeqIndexing(newSeqNode, seqNodeToMatch):
    """Sets indexing to be compatible by matching the index
    name, type, and units to match an existing sequence node.
    This is necessary for them to share a single sequence browser
    and be synchronized.
    """
    newSeqNode.SetIndexName(seqNodeToMatch.GetIndexName())
    newSeqNode.SetIndexType(seqNodeToMatch.GetIndexType())
    newSeqNode.SetIndexUnit(seqNodeToMatch.GetIndexUnit())


def cloneItem(dataNode):
    """Clone any data node in the scene subject hierarchy"""
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    itemID = shNode.GetItemByDataNode(dataNode)
    clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(
        shNode, itemID
    )
    clonedDataNode = shNode.GetItemDataNode(clonedItemID)
    return clonedDataNode


def markupsControlPointsToPolyData(markupsNode):
    """Control points are converted to the points of a vtkPolyLine and
    added to a vtkPolyData object
    """
    # Extract control points from the curve node
    control_points = vtk.vtkPoints()
    markupsNode.GetControlPointPositionsWorld(control_points)

    # Create a vtkPolyData object to store the control points
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(control_points)

    # Create a vtkPolyLine to connect the control points
    poly_line = vtk.vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(control_points.GetNumberOfPoints())
    for i in range(control_points.GetNumberOfPoints()):
        poly_line.GetPointIds().SetId(i, i)

    # Create a cell array to store the vtkPolyLine
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(poly_line)
    poly_data.SetLines(cells)

    # Now poly_data contains the vtkPolyData representation of the control points as a single polyline
    # print(f"Number of lines: {poly_data.GetNumberOfLines()}")
    return poly_data


def smoothPolyDataLine(polyData, smoothingFactor=0.2, numIterations=100):
    """With a vtkPolyData input composed of a single vtkPolyLine, the
    output is a smoothed version of the line.
    """
    try:
        import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
    except ImportError:
        raise ImportError("Please install VMTK extension")
    smoothing = vtkvmtkComputationalGeometry.vtkvmtkCenterlineSmoothing()
    # smoothing.SetInputConnection(reader.GetOutputPort())
    smoothing.SetInputData(polyData)
    smoothing.SetSmoothingFactor(smoothingFactor)
    smoothing.SetNumberOfSmoothingIterations(numIterations)
    smoothing.Update()
    smoothedPolyData = smoothing.GetOutput()
    return smoothedPolyData


def polyDataToMarkupsCurve(polyData, outputMarkupsNode=None):
    """Convert from a vtkPolyData containing a single line back to a
    markupsCurveNode with the points of the vtkPolyData as control
    points
    """
    points = polyData.GetPoints()
    if outputMarkupsNode is None:
        outputMarkupsNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "ConvertedCurve"
        )
    else:
        outputMarkupsNode.RemoveAllControlPoints()
    # Extract the vtkPolyLine from the poly_data
    lines = polyData.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    lines.GetNextCell(id_list)
    # This is the cell containing the single polyLine

    # Add the points to the curve node in the order specified by the vtkPolyLine
    points = polyData.GetPoints()
    for i in range(id_list.GetNumberOfIds()):
        point_id = id_list.GetId(i)
        point = points.GetPoint(point_id)
        outputMarkupsNode.AddControlPoint(vtk.vtkVector3d(point))
    return outputMarkupsNode


def smoothCurveNode(
    curveNode, smoothingFactor=0.2, numIterations=100, outputSmoothCurveNode=None
):
    """ """
    polyData = markupsControlPointsToPolyData(curveNode)
    smoothedPolyData = smoothPolyDataLine(
        polyData, smoothingFactor=smoothingFactor, numIterations=numIterations
    )
    outputSmoothCurveNode = polyDataToMarkupsCurve(
        smoothedPolyData, outputMarkupsNode=outputSmoothCurveNode
    )
    outputSmoothCurveNode.SetName(f"Smoothed_{curveNode.GetName()}")
    return outputSmoothCurveNode


def smoothPolydataSurf(polydata, numIter=10, relaxationFactor=0.2):
    """Returns a smoothed version of a polydata surface"""
    smoothFilter = vtk.vtkSmoothPolyDataFilter()
    smoothFilter.SetInputData(polydata)
    smoothFilter.SetNumberOfIterations(numIter)
    smoothFilter.SetRelaxationFactor(relaxationFactor)
    smoothFilter.BoundarySmoothingOn()
    smoothFilter.Update()
    return smoothFilter.GetOutput()


def findCarinaFromPolydata(polydata):
    # TODO: figure out how this works :)
    x = findCenterline(polydata)
    x1 = np.mean(x[:, 0]) - 7
    x2 = np.mean(x[:, 0]) + 7
    bounds = polydata.GetBounds()

    cutter = vtk.vtkPolyDataPlaneCutter()
    cutter.SetInputData(polydata)

    plane = vtk.vtkPlane()
    plane.SetNormal(1, 0, 0)
    max = bounds[4]
    k = x1
    for x in range(int(x2 - x1) * 10):
        plane.SetOrigin(x / 10 + x1, 0, 0)
        cutter.SetPlane(plane)
        cutter.Update()
        b = cutter.GetOutput().GetBounds()
        # fiducialNode.AddControlPoint(x/10, 0, b[4])
        if b[4] > max:
            max = b[4]
            k = x1 + x / 10

    plane.SetOrigin(k, 0, 0)
    cutter.SetPlane(plane)
    cutter.Update()

    pd = cutter.GetOutput()
    b = pd.GetBounds()
    for i in range(pd.GetNumberOfPoints()):
        if pd.GetPoint(i)[2] == b[4]:
            # fiducialNode.AddControlPoint(pd.GetPoint(i))
            break

    return pd.GetPoint(i)


# TODO no need to store them in a fiducial list, could be a np array
# Given a polydata, cut it in z direction with 1mm intervals
# Return the center of each "slice"
# Assumes somewhat upright trachea
def findCenterline(polydata):

    cutter = vtk.vtkPolyDataPlaneCutter()
    cutter.SetInputData(polydata)

    plane = vtk.vtkPlane()
    plane.SetNormal(0, 0, 1)

    # fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "F")
    points = list()
    bounds = polydata.GetBounds()
    for x in range(int(bounds[5] - bounds[4])):
        plane.SetOrigin(0, 0, x + bounds[4])
        cutter.SetPlane(plane)
        cutter.Update()
        # fiducialNode.AddControlPoint()
        points.append(cutter.GetOutput().GetCenter())

    # x = np.asarray(slicer.util.arrayFromMarkupsControlPoints(fiducialNode))
    # x = x[int(len(x) / 3):-1, :]
    # slicer.mrmlScene.RemoveNode(fiducialNode)
    points = np.asarray(points)
    return points


def rotationMatrix(normal1, normal2):
    vec = [0, 0, 0]
    vtk.vtkMath.Cross(normal1, normal2, vec)
    costheta = vtk.vtkMath.Dot(normal1, normal2)
    sintheta = vtk.vtkMath.Norm(vec)
    theta = np.arctan2(sintheta, costheta)
    if sintheta != 0:
        vec[0] = vec[0] / sintheta
        vec[1] = vec[1] / sintheta
        vec[2] = vec[2] / sintheta
    # convert to quaternion
    costheta = np.cos(0.5 * theta)
    sintheta = np.sin(0.5 * theta)
    quat = [costheta, vec[0] * sintheta, vec[1] * sintheta, vec[2] * sintheta]
    # convert to matrix
    mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    vtk.vtkMath.QuaternionToMatrix3x3(quat, mat)

    vtkmat = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            vtkmat.SetElement(i, j, mat[i][j])

    return vtkmat


def findVector(obs_points):
    C_x = np.cov(
        obs_points.T
    )  # Note that here numpy does row-order to find covariance.
    # If we don't do it this way, we'll get an 8 x 8 matrix
    # instead of a 3 x 3 matrix.

    # Note we're using eigh below, not eig. This is because C_x is a symmetrical (also
    # called Hermitian) matrix, so eigh (eigen decomposition for hermitian matrix) is
    # more appropriate
    eig_vals, eig_vecs = np.linalg.eigh(C_x)

    variance = np.max(eig_vals)
    max_eig_val_index = np.argmax(eig_vals)
    direction_vector = eig_vecs[:, max_eig_val_index].copy()
    return direction_vector


def rotatePolydata(polydata, normal2):
    center = polydata.GetCenter()
    centeringTransform = vtk.vtkTransform()
    centeringTransform.Translate(-center[0], -center[1], -center[2])
    centeringTransform.Update()

    normal1 = (0, 0, 1)

    mat = rotationMatrix(normal2, normal1)
    rotation = vtk.vtkTransform()
    rotation.SetMatrix(mat)
    rotation.Update()

    decenteringTransform = vtk.vtkTransform()
    decenteringTransform.Translate(center)
    decenteringTransform.Update()

    decenteringTransform.Concatenate(rotation)
    decenteringTransform.Concatenate(centeringTransform)
    decenteringTransform.Update()

    return decenteringTransform


def convertSegmentToModelNode(segmentID, segmentationNode, outputModelNode=None):
    # Convert segment with given ID in given segmentation node to model node representation
    # TODO: handle bad inputs appropriately
    segmentIdx = segmentationNode.GetSegmentation().GetSegmentIndex(segmentID)
    segment = segmentationNode.GetSegmentation().GetNthSegment(segmentIdx)
    if outputModelNode is None:
        outputModelName = "_".join((segmentID, "ModelNode"))
        outputModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", outputModelName
        )
    slicer.modules.segmentations.logic().ExportSegmentToRepresentationNode(
        segment, outputModelNode
    )
    return outputModelNode


def transformPolydata(polydata, transform):
    filter = vtk.vtkTransformPolyDataFilter()
    filter.SetInputData(polydata)
    filter.SetTransform(transform)
    filter.Update()
    return filter.GetOutput()


def alignAirwayToTracheaCenterline(polydata):
    centerlinePoints = findCenterline(polydata)
    normal = findVector(centerlinePoints)

    # If the vector is pointing downwards, flip
    if normal[2] < 0:
        normal[0] = -normal[0]
        normal[1] = -normal[1]
        normal[2] = -normal[2]

    transform = rotatePolydata(polydata, normal)
    newPolydata = transformPolydata(polydata, transform)

    return newPolydata, transform


def getTransformSequenceFromLandmarksSequence(
    landmarkSeqNode, outputTranslationTformSeq=None, addToBrowser=False
):
    """From carina landmarks sequence, create translation-only transforms which
    align the carina points.
    """
    scene = slicer.mrmlScene
    browserNode = getFirstBrowser(landmarkSeqNode)
    # Get first landmark location (all transforms should take the landmark to
    # this point)
    browserNode.SetSelectedItemNumber(0)
    landmarkProxy = browserNode.GetProxyNode(landmarkSeqNode)
    firstLandmarkLocation = np.array(landmarkProxy.GetNthControlPointPosition(0))

    if outputTranslationTformSeq is None:
        # Create new sequence to hold transforms
        outputTranslationTformSeq = scene.AddNewNodeByClass(
            "vtkMRMLSequenceNode", "TranslationTransformsSeq"
        )
    outputTranslationTformSeq.CopySequenceIndex(landmarkSeqNode)
    for frameIdx in range(browserNode.GetNumberOfItems()):
        browserNode.SetSelectedItemNumber(frameIdx)
        curLandmarkLocation = np.array(landmarkProxy.GetNthControlPointPosition(0))
        rasShift = firstLandmarkLocation - curLandmarkLocation
        trNode = createOrUpdateTranslationTransform(rasShift)
        # Add to transform sequence
        indexValue = landmarkSeqNode.GetNthIndexValue(frameIdx)
        outputTranslationTformSeq.SetDataNodeAtValue(trNode, indexValue)
        scene.RemoveNode(trNode)
    if addToBrowser:
        # Add completed sequence to the browser node
        seqID = outputTranslationTformSeq.GetID()
        if not browserNode.IsSynchronizedSequenceNodeID(seqID):
            browserNode.AddSynchronizedSequenceNodeID(seqID)
            browserNode.SetSaveChanges(outputTranslationTformSeq, True)
            # browserNode.SetOverwriteProxyName(transformSeq, True)
    return outputTranslationTformSeq


def createOrUpdateTranslationTransform(RasShift: np.ndarray, outputTransformNode=None):
    """Given an RAS shift amount, the given transform node has its matrix
    set to be the 4x4 transform matrix with identity in the first 3 columns
    and the RAS shift amount as the translation component. If no transform
    node is supplied, one is created and returned. RasShift must have exactly
    three elements.
    """
    scene = slicer.mrmlScene
    trMatrix = np.eye(4)
    trMatrix[0:3, 3] = (*RasShift,)
    if outputTransformNode is None:
        transformName = scene.GenerateUniqueName("TranslationMatrix")
        outputTransformNode = scene.AddNewNodeByClass(
            "vtkMRMLLinearTransformNode", transformName
        )
    slicer.util.updateTransformMatrixFromArray(outputTransformNode, trMatrix)
    return outputTransformNode


def keep_largest_island(segmentID, segmentationNode):
    """Keep only the largest island of segment"""
    import SegmentEditorEffects

    segmentEditorWidget, segmentEditorNode, segmentationNode = setup_segment_editor(
        segmentationNode
    )
    segmentEditorNode.SetSelectedSegmentID(segmentID)
    segmentEditorWidget.setActiveEffectByName("Islands")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", SegmentEditorEffects.KEEP_LARGEST_ISLAND)
    effect.setParameter(
        "BypassMasking", 1
    )  # will not modify any other segments or apply any intensity masking to the results
    effect.self().onApply()
    # Fix cursor staying stuck on islands tool icon
    segmentEditorWidget.setActiveEffectByName("")
    slicer.mrmlScene.RemoveNode(segmentEditorNode)


def fillROISegment(roiNode, segmentationNode, segmentName="FilledROI"):
    """Create a segment which corresponds to the inside of the
    ROI node.  Uses Surface Cut tool of segment editor to construct.
    """
    # Get corner points of roi
    cornerPoints = getROICornerPoints(roiNode)
    # Set up editor
    segmentEditorWidget, segmentEditorNode, segmentationNode = setup_segment_editor(
        segmentationNode, None
    )
    # Add segment
    segmentID = segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
    segmentEditorNode.SetSelectedSegmentID(segmentID)
    # Activate surface cut effect and retrieve it
    segmentEditorWidget.setActiveEffectByName("Surface cut")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("Operation", "FILL_INSIDE")
    # Place input boundary points
    effect.self().fiducialPlacementToggle.placeButton().click()
    tempMarkup = effect.self().segmentMarkupNode
    for p in cornerPoints:
        tempMarkup.AddFiducialFromArray(p)
    effect.self().fiducialPlacementToggle.placeButton().click()  # turn off fiducial placement mode
    effect.self().smoothModelCheckbox.setChecked(False)  # use flat sides
    # Masking Settings
    segmentEditorNode.SetMaskSegmentID(None)
    segmentEditorNode.SetMaskMode(segmentationNode.EditAllowedEverywhere)
    segmentEditorNode.MasterVolumeIntensityMaskOff()
    segmentEditorNode.SetOverwriteMode(segmentEditorNode.OverwriteNone)
    # Run the thresholded surface cut segmentation
    effect.self().onApply()
    # Clean up
    effect.self().observeSegmentation(
        False
    )  # otherwise observer for SegmentModified event hangs around!
    slicer.mrmlScene.RemoveNode(segmentEditorNode)
    slicer.mrmlScene.RemoveNode(tempMarkup)
    # for node in slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode"):
    #    if node.GetName() == "C":
    #        slicer.mrmlScene.RemoveNode(node)
    slicer.mrmlScene.RemoveNode(slicer.util.getNode("SegmentEditorSurfaceCutModel"))
    return segmentID


def getROICornerPoints(roiNode, world=True) -> List[List[float]]:
    """Gather the coordinates of the corners of the roiNode."""
    objectCorners = []
    sideLengths = roiNode.GetSize()
    xSize, ySize, zSize = sideLengths
    for x in [-0.5 * xSize, 0.5 * xSize]:
        for y in [-0.5 * ySize, 0.5 * ySize]:
            for z in [-0.5 * zSize, 0.5 * zSize]:
                objectCorners.append([x, y, z])
    if not world:
        # Incorporate any internal rotation associated with the node
        objToNode = slicer.util.arrayFromVTKMatrix(roiNode.GetObjectToNodeMatrix())
        nodeCorners = []
        for objectCorner in objectCorners:
            objectCornerH = np.array([*objectCorner, 1])
            nodeCornerH = objToNode @ np.array(objectCornerH)
            nodeCorner = list(nodeCornerH[:3])
            nodeCorners.append(nodeCorner)
        corners = nodeCorners
    else:
        # if world, use object to world matrix instead
        objToWorld = slicer.util.arrayFromVTKMatrix(roiNode.GetObjectToWorldMatrix())
        worldCorners = []
        for objectCorner in objectCorners:
            objectCornerH = np.array([*objectCorner, 1])
            worldCornerH = objToWorld @ np.array(objectCornerH)
            worldCorner = list(worldCornerH[:3])
            worldCorners.append(worldCorner)
        corners = worldCorners
    # NOTE on how Slicer handles ROIs under possibly nonlinear transforms
    # Slicer applies any set of concatenated parent transforms to the
    # CENTER point of the ROI and to the AXIS DIRECTIONS of the ROI. Then
    # The transformed axes are made orthogonal again by sequentially
    # applying cross products (z is made perpendicular to x and y, then
    # y is made perpendicular to x and z, and then x is made perpendicular
    # to y and z (it should be already, but this process ensures handedness
    # and also does normalization)).  Applied scaling is also tracked and
    # stored in a new size, accessible via GetSizeWorld.  The transformed
    # ROI box you see is constructed using the world size, and the new (but
    # still orthonormal) axis directions. Therefore, ROIs are always still
    # rectilinear boxes, even under shear or nonlinear warping transforms.
    # Because of this, it does not work to simply apply parent transforms
    # to ROI node corner locations, because these may end up in non-
    # rectilinear configurations.  If you want something which starts
    # from a rectilinear ROI but deforms in a detailed way, it might be
    # a better approach to make a segment or labelmap from the initial
    # ROI and apply the parent transforms to that, and harden.
    return corners


def getAspectRatiosAndAreas(smoothCenterlineNode, segmentationNode, segmentIdx=0):
    """
    NO LONGER USED, REFACTORED AWAY
    Calculate aspect ratios (long axis, short axis, aspect ratio)
    and cross-section area, of slices of the given segment in the given
    segmentation node, perpendicular to the smoothed centerline at each of
    the control points of the smoothed centerline.
    """
    # segID = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segName)
    segID = segmentationNode.GetSegmentation().GetNthSegmentID(segmentIdx)
    segPolyData = vtk.vtkPolyData()
    segmentationNode.GetClosedSurfaceRepresentation(segID, segPolyData)

    surf = pv.wrap(segPolyData)
    cpArr = arrayFromMarkupsControlPoints(smoothCenterlineNode)
    sliceNormals = cpArr[1:, :] - cpArr[:-1, :]
    # repeat last normal for last point
    sliceNormals = np.vstack((sliceNormals, sliceNormals[-1, :]))
    dataDicts = []
    for cp, normal in zip(cpArr, sliceNormals):
        clip = surf.slice(normal=normal, origin=cp)
        clip_connected = clip.connectivity("closest", closest_point=cp)
        dataDict = compute_aspect_ratio(clip_connected, normal)
        dataDict["CSA"] = compute_area(clip_connected)
        dataDicts.append(dataDict)
    return dataDicts


# Refactor getAspectRatiosAndAreas to avoid repeating calculations unnecessarily
# The centerline points and normals are the same for every frame, just calculate
# them once. centerlinePoints, centerlineNormals.  For each frame, there is
# one clip_connected for every point/normal/frame combination.  The aspect
# ratio data and the CSA are each calculated from the clip_connected.

"""
Currently, runQuantification does 
"""


def getCenterlineSlicePointsAndNormals(
    centerline: vtkMRMLMarkupsCurveNode,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    slicePoints = arrayFromMarkupsControlPoints(centerline)
    sliceNormals = slicePoints[1:, :] - slicePoints[:-1, :]
    # repeat last normal for last point
    sliceNormals = np.vstack((sliceNormals, sliceNormals[-1, :]))
    return slicePoints, sliceNormals


def find_intersection_points(polyline: pv.PolyData, planeNormal, planePoint):
    """Given a polyline curve (pyvista wrapped vtkPolyData containing
    a single vtkPolyLine), find any intersection points with a
    supplied plane defined by a normal vector and a point on
    the plane.  The expected case is two intersections, but, for
    corner cases there may be more than two.  The returned points
    are ordered so that the first two can hopefully be used as a
    short axis. To accomplish this, points are ordered by distance
    to the planePoint.  A failure mode could be if there are two
    intersections near the planePoint and another on the other side
    but further away.  That will be addressed if needed by future
    modifications.
    Returns None if no intersections found, otherwise returns
    the intersection points as
    """
    if polyline.n_points == 0:
        return None
    # all_edges = polyline.extract_all_edges()
    intersection_points = []
    points = polyline.points  # all_edges.points
    # cells = all_edges.lines.reshape(-1, 3)[:, 1:3]  # extract edge indices
    cells = polyline.lines.reshape(-1, 3)[:, 1:3]
    for cell in cells:
        p1 = points[cell[0]]
        p2 = points[cell[1]]
        # Plane equation: normal . (point - origin) = 0
        # Line equation: point = p1 + t * (p2 - p1)
        line_dir = p2 - p1
        plane_normal = np.array(planeNormal)
        plane_origin = np.array(planePoint)
        denom = np.dot(line_dir, plane_normal)
        if abs(denom) > 1e-8:  # Avoid division by zero
            t = np.dot(plane_origin - p1, plane_normal) / denom
            if 0 <= t <= 1:
                intersection_point = p1 + t * line_dir
                intersection_points.append(intersection_point)
        # Skips any edge parallel to the slice plane

    if intersection_points:
        # Order by increasing distance from planePoint
        intersection_points = np.array(intersection_points)
        origin = np.array(plane_origin)
        distances = np.linalg.norm(intersection_points - origin, axis=1)
        sorted_indices = np.argsort(distances)
        return intersection_points[sorted_indices]
    else:
        return None


def compute_aspect_ratio(clip, norm):
    """clip: contour/line defining clipped surface
    norm: normal vector of plane used to clip surface
    """
    hdist = cdist(clip.points, clip.points, metric="euclidean")

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    # Define line between two points
    long_axis = pv.Line(clip.points[bestpair[0]], clip.points[bestpair[1]])
    long_axis_center = np.array(long_axis.center)
    long_axis_vector = long_axis.points[1] - long_axis.points[0]
    # Define the plane the short axis must lie in
    short_axis_plane_normal = long_axis_vector
    # Find the intersection points of the short axis plane with the
    # clip curve
    intersection_points = find_intersection_points(
        clip, short_axis_plane_normal, long_axis_center
    )
    if intersection_points is None:
        raise RuntimeError(
            "No intersection found between short axis plane and clipped airway surface.  This shouldn't be possible"
        )
    elif intersection_points.shape[0] > 2:
        logging.warning(
            "More than two points found along short axis intersection plane with clip contour, only the two closest to the long axis center were used, but there could be problems!"
        )
    elif intersection_points.shape[0] == 2:
        # We're good
        pass
    else:
        raise RuntimeError(
            "find_intersection_points returned some kind of impossible result"
        )
    # Short axis connects the first two intersection points
    short_axis = pv.Line(intersection_points[0], intersection_points[1])
    # Organize outputs
    outputDict = {
        "aspectRatio": short_axis.length / long_axis.length,
        "longAxisLength": long_axis.length,
        "shortAxisLength": short_axis.length,
        "longAxisPoints": np.array(long_axis.points),
        "shortAxisPoints": np.array(short_axis.points),
    }
    return outputDict


def compute_aspect_ratio_OLD(clip, norm, debug=False):
    """
    clip: contour/line defining clipped surface
    norm: normal vector of plane used to clip surface
    """
    hdist = cdist(clip.points, clip.points, metric="euclidean")

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    # Define line between two points
    long_axis = pv.Line(clip.points[bestpair[0]], clip.points[bestpair[1]])
    long_axis_center = np.array(long_axis.center)

    # translate clip and lines to origin
    long_axis.translate(long_axis_center * -1, inplace=True)
    clip.translate(long_axis_center * -1, inplace=True)

    # compute rotation to align max dimension with y-axis and clip plane with z axis
    max_dim_vec = long_axis.points[1] - long_axis.center
    norm_max_dim_vec = max_dim_vec / np.linalg.norm(max_dim_vec)

    A = [norm_max_dim_vec.tolist(), norm]
    axis = [[0, 1, 0], [0, 0, 1]]

    T = np.zeros((4, 4))
    rot, _ = Rotation.align_vectors(axis, A)
    T[0:3, 0:3] = rot.as_matrix()
    T[3, 3] = 1

    # rotate clipped surface and long axes
    clip_rot = clip.transform(T, inplace=False)
    long_axis_rot = long_axis.transform(T, inplace=False)

    # find the length of curve in x-direction
    #### this was more of a pain than i thought. clipping a curve with a line/plane is a challenge ###
    ## if you're using clipped surfaces and not boundaries/curves there's likely a much easier way to do this ####

    max_x_size = abs(abs(clip_rot.bounds[1] - clip_rot.bounds[0]))
    x_clip_Plane = pv.Plane(
        center=long_axis_rot.center,
        direction=[0, 1, 0],
        i_size=max_x_size * 2,
        j_size=max_x_size * 2,
    )

    half_curve = clip_rot.clip(origin=long_axis_rot.center, normal="y")
    closest_cells, closest_points = half_curve.find_closest_cell(
        x_clip_Plane.points, return_closest_point=True
    )
    distance = np.linalg.norm(x_clip_Plane.points - closest_points, axis=1)
    pts_id = np.argsort(distance)[0:2]
    end1_curve = half_curve.points[
        half_curve.find_closest_point(x_clip_Plane.points[pts_id[0]])
    ]
    end2_curve = half_curve.points[
        half_curve.find_closest_point(x_clip_Plane.points[pts_id[1]])
    ]
    small_axes = pv.Line(end1_curve, end2_curve)

    if debug:
        p = pv.Plotter()
        p.add_mesh(clip)
        p.add_mesh(clip_rot, color="red", label="AR_curve")
        p.add_mesh(long_axis)
        p.add_mesh(small_axes, color="red")
        p.add_mesh(long_axis_rot, color="red")
        p.add_legend()
        p.add_axes()
        p.show()
    long = long_axis.length
    short = small_axes.length

    aspect_ratio = {"Long Axis": long, "Short Axis": short, "AR": short / long}

    return aspect_ratio


def compute_area(clip):
    """Input clip should be a single, in-plane contour polydata"""
    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(clip)
    triangulator.Update()
    mass_properties = vtk.vtkMassProperties()
    mass_properties.SetInputData(triangulator.GetOutput())
    area = mass_properties.GetSurfaceArea()
    return area


def getFirstBrowser(seqNode):
    browserNode = slicer.modules.sequences.logic().GetFirstBrowserNodeForSequenceNode(
        seqNode
    )
    return browserNode


def removeSequence(seqNode, removeProxy=True, removeBrowser=False):
    """Convenience function to remove a sequence node from the scene
    (because it is a pain in the butt to do using the GUI)
    Note, this assumes that a browser exists for the sequence, that
    there is only one browser for the sequence, and that there is
    only one proxy node for the sequence. This is the typical case,
    but Slicer allows other situations.  If you are doing something
    more complex, you will need to handle it.
    The removal has to happen in a particular order: sequence removal,
    then proxyNode removal, then browser removal. However, since
    we need the sequence to obtain the browser and the browser and
    the sequence to obtain the proxyNode, we have to gather everything
    first, and then start removal.
    """
    scene = slicer.mrmlScene
    # Gather browser and proxy node
    browserNode = getFirstBrowser(seqNode)
    if browserNode and removeProxy:
        proxyNode = browserNode.GetProxyNode(seqNode)
    # Something gets confused if I run this, and a new sequence is generated,
    # maybe it will help if we remove the sequence node from the browser first
    browserNode.RemoveSynchronizedSequenceNode(seqNode.GetID())
    # Remove requested elements
    scene.RemoveNode(seqNode)
    if browserNode and removeProxy:
        scene.RemoveNode(proxyNode)
    if removeBrowser:
        scene.RemoveNode(browserNode)


def tableNodeFromArray(
    arr: np.ndarray, colNamesOrPrefix="Col", transpose=False, outputTableNode=None
):
    """Creates or updates a vtkMRMLTableNode with data from a numpy ndarray
    with two dimensions.  Column names (which will appear as column headers
    for the table node) must be supplied.  The array will be transposed
    before taking the columns if transpose is True (default False).  If
    and outputTableNode is supplied, it is cleared out and rebuilt from
    the current array data. If no outputTableNode is supplied, then a
    new table node is created and returned.
    """
    scene = slicer.mrmlScene
    if transpose:
        arr = arr.transpose()
    nCols = arr.shape[1]
    if type(colNamesOrPrefix) == str:
        # prefix supplied, builld column names from that
        prefix = colNamesOrPrefix
        colNames = tuple(f"{prefix} #{idx}" for idx in range(nCols))
    elif len(colNamesOrPrefix) == nCols:
        # each column name supplied
        colNames = colNamesOrPrefix
    else:
        raise RuntimeError(
            f"Either a column prefix or a list of column names must be supplied, cannot create table node!"
        )

    if not outputTableNode:
        tableNode = scene.AddNewNodeByClass("vtkMRMLTableNode")
    else:
        # Clear out any existing data
        tableNode = outputTableNode
        tableNode.RemoveAllColumns()
    # Create columns and add to table
    for colIdx in range(nCols):
        col = vtk.vtkFloatArray()
        col.SetName(colNames[colIdx])
        for val in arr[:, colIdx]:
            col.InsertNextValue(val)
        tableNode.GetTable().AddColumn(col)
    return tableNode


def tableSeqFromArray(dataArr, xCol, browserNode, dataLabel):
    """From a 2d numpy array with time in the columns, create
    a sequence of table nodes (one data column in each table)
    Also add the supplied column (xCol) to each table (so it
    can serve as the x coordinate for plot series).  A
    browser node you want to sync with is also required to
    set index values.
    """
    dynTableSeq = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSequenceNode", f"dyn{dataLabel}TableSeq"
    )
    otherSeq = browserNode.GetMasterSequenceNode()
    setCompatibleSeqIndexing(dynTableSeq, otherSeq)
    for colIdx in range(dataArr.shape[1]):
        colData = dataArr[:, colIdx]
        tempTableNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTableNode", f"{dataLabel} {colIdx} Table"
        )
        vCol = vtk.vtkFloatArray()
        vCol.SetName(dataLabel)
        for csa in colData:
            vCol.InsertNextValue(csa)
        tempTableNode.AddColumn(vCol)
        tempTableNode.AddColumn(xCol)
        # Add to sequence
        indexValue = otherSeq.GetNthIndexValue(colIdx)
        dynTableSeq.SetDataNodeAtValue(tempTableNode, indexValue)
        # Clean up
        slicer.mrmlScene.RemoveNode(tempTableNode)
    # Add to browser node
    browserNode.AddSynchronizedSequenceNode(dynTableSeq)
    browserNode.SetSaveChanges(dynTableSeq, True)
    return dynTableSeq


"""
To make plots, I need plotSeries objects, which need to be linked
to table nodes.  That means I need to have any envelope curves
ALSO put into a table.  That table needs to have the x column (carina
distances), temporal max, temporal min, and maybe centerline point
index columns. The current slice can be marked with a vertical line.

"""


def makeEnvTable(
    dataArr,
    transposeFlag: bool,
    distances,
    dataLabel="CSA",
    centerlineIdxs=None,
    outputTableNode=None,
):
    """Make the supporting data table for plot series for an
    envelope plot.  Required columns are DistanceToCarina_Mm,
    {dataLabel}_Max, {dataLabel}_Min, and CenterlinePointIndex
    By default, centerlineIdxs are assumed to be the same a just
    numbering the carina distance, but the input is included here
    because I'm not sure yet how I want to handle trimming
    the data range for figures or exports, and this might be
    one place to do it.
    """
    if transposeFlag:
        dataArr = np.transpose(dataArr)
    if not dataArr.shape[0] == distances.shape[0]:
        raise ValueError(
            "Data array must have the same number of rows as there are carina distances!"
        )
    # Find max/min over time
    lowestData = np.min(dataArr, axis=1)
    highestData = np.max(dataArr, axis=1)
    # Gather each column
    minCol = makeFloatCol(f"Min{dataLabel}", lowestData)
    maxCol = makeFloatCol(f"Max{dataLabel}", highestData)
    distCol = makeFloatCol(f"DistToCarinaMm", distances)
    if centerlineIdxs is None:
        centerlineIdxs = np.array((range(distCol.GetNumberOfValues())))
    clineIdxCol = makeFloatCol(f"CenterlineIdx", centerlineIdxs)
    # Assemble into table node
    scene = slicer.mrmlScene
    if not outputTableNode:
        outputTableNode = scene.AddNewNodeByClass("vtkMRMLTableNode")
        uname = scene.GenerateUniqueName(f"{dataLabel}EnvTable")
        outputTableNode.SetName(uname)
    outputTableNode.RemoveAllColumns()
    outputTableNode.AddColumn(clineIdxCol)
    outputTableNode.AddColumn(distCol)
    outputTableNode.AddColumn(minCol)
    outputTableNode.AddColumn(maxCol)
    return outputTableNode


def makeFloatCol(colName: str, colData: np.ndarray):
    """Make vtkFloatArray to serve as table column."""
    col = vtk.vtkFloatArray()
    col.SetName(colName)
    if len(colData.shape) == 1:
        colData = colData.reshape((-1, 1))
    nComponents = colData.shape[1]
    nRows = colData.shape[0]
    for idx in range(nRows):
        if nComponents == 1:
            col.InsertNextValue(colData[idx])
        else:
            col.InsertTuple(idx, colData[idx, :])
    return col


def makeRangeCol(colName: str, data):
    """Make a vtkFloatArray to serve as a table column.
    It will have two entries, the maximum of the data
    and the minimum of the data
    """
    colData = np.array((np.max(np.array(data)[:]), np.min(np.array(data)[:])))
    col = makeFloatCol(colName, colData)
    return col


def showPlot(chartNode):
    showPlotLayoutNum = 36
    slicer.app.layoutManager().setLayout(showPlotLayoutNum)
    plotViewNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLPlotViewNode")
    plotViewNode.SetPlotChartNodeID(chartNode.GetID())


def serialize_dict_with_ndarrays(data_dict):
    """
    Serializes a dictionary with string keys and numpy.ndarray values to a JSON string.

    Args:
        data_dict (dict): The dictionary to serialize.

    Returns:
        str: The JSON string representation of the dictionary.
    """
    serializable_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            serializable_dict[key] = value.tolist()  # Convert ndarray to list
        else:
            serializable_dict[key] = value  # Handle non-ndarray values if necessary

    return json.dumps(serializable_dict)


def deserialize_dict_with_ndarrays(json_string):
    """
    Deserializes a JSON string back into a dictionary with numpy.ndarray values.

    Args:
        json_string (str): The JSON string to deserialize.

    Returns:
        dict: The deserialized dictionary.
    """
    deserialized_dict = json.loads(json_string)
    for key, value in deserialized_dict.items():
        if isinstance(value, list):
            deserialized_dict[key] = np.array(value)  # Convert list back to ndarray
    return deserialized_dict


def removeNthSegmentFromSequence(segSeq, idxToRemove):
    """Convenience function to remove extra segments which
    have been added to a segmentation sequence.
    For example if re-running volume quantification.
    """
    browser = getFirstBrowser(segSeq)
    nFrames = segSeq.GetNumberOfDataNodes()
    segNode = browser.GetProxyNode(segSeq)
    for idx in range(nFrames):
        browser.SetSelectedItemNumber(idx)
        vSeg = segNode.GetSegmentation()
        nSegments = vSeg.GetNumberOfSegments()
        lastSegIdx = nSegments - 1
        # skip any frame where there is no nth segment
        if idxToRemove <= lastSegIdx:
            segmentToRemove = vSeg.GetNthSegment(idxToRemove)
            vSeg.RemoveSegment(segmentToRemove)
