import cv2
from lib.coordinate_transform import windowToFieldCoordinates
from lib.average_coordinates import getRunningAverageCoordinates

class Heatmap:
	def __init__(self, frame, fieldWidth, fieldHeight):
		self.frame = frame
		self.fieldWidth = fieldWidth
		self.fieldHeight = fieldHeight

	def getPosRelativeCoordinates(self, originalPoint, perspectiveCoords):
		
		resultCoord = windowToFieldCoordinates(originalPoint, perspectiveCoords, self.fieldWidth, self.fieldHeight)
		x = int(resultCoord[0])
		y = int(resultCoord[1])

		return (x, y)

	def getPosAbsoluteCoordinates(self, posRelative, fieldTopLeftPoint):
		
		(x1, y1) = fieldTopLeftPoint
		posRelativeAvg = getRunningAverageCoordinates(posRelative)
		x = x1 + int(posRelativeAvg[0])
		y = y1 + int(posRelativeAvg[1])

		return (x, y)

	def drawOpacityCircle(self, position, colorR, colorG, colorB, radius, thickness):
		
		overlay = self.frame.copy()
		cv2.circle(overlay, position, radius, (colorB, colorG, colorR), thickness)
		alpha = 0.25
		cv2.addWeighted(overlay, alpha, self.frame, 1 - alpha, 0, self.frame)
