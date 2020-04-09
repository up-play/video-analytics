import cv2

class Mouse:
	x = 0
	y = 0
	rightClicked = False
	leftClicked = False

	def leftClick(self,event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.x = x
			self.y = y
			self.leftClicked = True