import cv2

def getPerpectiveCoordinates(image, windowName, mouse):
	
	cv2.imshow(windowName, image)	
	cv2.setMouseCallback(windowName, mouse.leftClick)
	
	i = 0
	coords = []
	
	while i < 4:
		i += 1
		mouse.x = 0
		mouse.y = 0
		mouse.leftClicked = False
		
		while mouse.leftClicked == False:
			# wait for key press
			key = cv2.waitKey(1) & 0xFF
		
		coords.append((mouse.x, mouse.y))

	return coords