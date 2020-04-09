import cv2

def drawQuadrilateral(image, pointsCoords, colorR, colorG, colorB, lineThickness):
	
	cv2.line(image, pointsCoords[0], pointsCoords[1], (colorB, colorG, colorR), lineThickness)
	cv2.line(image, pointsCoords[1], pointsCoords[2], (colorB, colorG, colorR), lineThickness)
	cv2.line(image, pointsCoords[2], pointsCoords[3], (colorB, colorG, colorR), lineThickness)
	cv2.line(image, pointsCoords[3], pointsCoords[0], (colorB, colorG, colorR), lineThickness)