# from dqn_globals import *
import numpy as np
import random
import time
import sys
#from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib.patches as patches
# from matplotlib import animation
import cv2
from environment import Environment

VERBOSE = False
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
VISUAL_DELAY = 0.075
VISUALIZE = False

class TetrisPiece:

    def __init__(self, block_type="leftl", column=0):
        self.column = column
        if block_type == "line":
            self.blocks = np.array([[1, 1, 1, 1]])
        elif block_type == "leftl":
            self.blocks = np.array([[1, 1, 1],[0, 0, 1]])
        elif block_type == "rightl":
            self.blocks = np.array([[0, 0, 1],[1, 1, 1]])
        elif block_type == "square":
            self.blocks = np.array([[1,1],[1,1]])
        elif block_type == "s":
            self.blocks = np.array([[0,1,1],[1,1,0]])
        elif block_type == "z":
            self.blocks = np.array([[1,1,0],[0,1,1]])
        elif block_type == "t":
            self.blocks = np.array([[0,1,0],[1,1,1]])

    def __str__(self):
        return str(self.blocks)

    def rotate(self, number_rots):
        tp = TetrisPiece()
        tp.setBlocks(np.rot90(self.blocks, number_rots))
        return tp

    def setColumn(self, column):
        self.column = column

    def setBlocks(self, blocks):
        self.blocks = blocks

    def getWidth(self):
        return self.blocks.shape[1]

    def getHeight(self):
        return self.blocks.shape[0]

    def getBottom(self, offset):
        for r in range(self.getHeight()-1, -1, -1):
            if (self.blocks[r][offset] != 0):
                return r;
        return 0;


class SimpleTetrisEnvironment(Environment):

    # board_width = FRAME_HEIGHT
    # board_height = FRAME_WIDTH

    def __init__(self, width, height, num_actions):
        self.board_height = height
        self.board_width = width
        self.num_actions = num_actions
        self.game_moves = 0
        self.frame_count = 0   # for visualization
        # self.games_file = open("current_games.txt", 'w')
        self.reset()


    def getRandomPiece(self):
        piece_type = random.choice(['square', 'leftl', 'rightl', 't', 's', 'z', 'line'])
        #piece_type = random.choice(['square'])
        # piece_type = random.choice(['line'])
        tp = TetrisPiece(piece_type)
        tp.column = self.board_width / 2
        return tp

    def getActionSet(self):
        return range(self.num_actions)  # 4 rotation, 10 starting columns

    def addPieceToBoard(self, block_value=1.0):
        # print "OLD BOARD:"
        # self.printBoard()
        # print "BLOCK TYPE:", self.current_piece.blocks
        c = self.current_piece.column
        for w in range(self.current_piece.getWidth()):
            for h in range(self.current_piece.getHeight()):
                if (self.current_piece.blocks[h][w] != 0):
                    #print "BORAD:", self.board, w, h
                    self.board[h][c+w] = block_value

        # print "NEW BOARD:"
        # self.printBoard()


    def copyPieceToLocation(self, row, col):
        #print "COPYING TO:", row, col
        for r in range(self.current_piece.getHeight()):
            for c in range(self.current_piece.getWidth()):
                if (self.current_piece.blocks[r][c] != 0):
                    # print self.current_piece, row, r, col, c
                    self.board[row+r][col+c] = 1.0


    def removePieceFromBoard(self):
        self.addPieceToBoard(0)

    def getScreen(self):
        #self.addPieceToBoard()
        return self.board

    def isLegalMove(self, action):
        rotation, position = self.actionToRotPos(action)
        rotated = self.current_piece.rotate(rotation)
        rotated.setColumn(position)
        # print "ISLEGAL:", rotated
        return rotated.column >= 0 and rotated.column + rotated.getWidth() - 1 < self.board_width

    def dropCurrentPiece(self):
        for row in range(self.board_height):
            for offset in range(self.current_piece.getWidth()):
                ledge = row + self.current_piece.getBottom(offset) + 1;
                # print self.current_piece, self.current_piece.column, offset
                if (ledge >= self.board_height or self.board[ledge][self.current_piece.column + offset] != 0):
                    self.copyPieceToLocation(row, self.current_piece.column)
                    return row > 2  # top two rows are for the incoming piece
        return False

    def actionToRotPos(self, action):
        # extract rotation and position
        return action / self.board_width, action % self.board_width

    def checkEliminate(self, row):
        for col in range(self.board_width):
            if (self.board[row][col] == 0):
                return False
        return True

    def clearRows(self):
        rows_cleared = 0
        for row in range(self.board_height):
            if (self.checkEliminate(row)):
                self.eliminateRow(row)
                rows_cleared += 1
        return rows_cleared

    def aboveFallsDown(self, row, col):
        for current_row in range(row, -1, -1):
            self.board[current_row][col] = self.board[current_row-1][col];
        self.board[0][col] = 0;

    def eliminateRow(self, row):
        for col in range(self.board_width):
            self.board[row][col] = 0;
            fallRow = row
            while (fallRow+1 < self.board_height and self.board[fallRow+1][col] == 0):
                fallRow += 1
            self.aboveFallsDown(fallRow, col)

    def visualizeBoard(self, filename):
        img = np.copy(self.board)

        img *= 255.
        res = cv2.resize(img, None, fx=40, fy=40, interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(filename, res)

        #res = cv2.resize(img, None, fx=40, fy=40, interpolation = cv2.INTER_NEAREST)
        cv2.imshow('image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(0)

        time.sleep(VISUAL_DELAY)

    def printBoard(self):
        print self.getBoardString()

    def getBoardString(self):
        board_string = ''
        for row in range(self.board_height):
            for col in range(self.board_width):
                board_string += "%d " % self.board[row][col]
            board_string += "\n"
        board_string += "_" * 40
        board_string += "\n"
        return board_string


    def performRandomAction(self):
        self.performAction(random.choice(self.getActionSet()))

    def performAction(self, action):

        self.game_moves += 1
        if not self.isLegalMove(action):
            if VERBOSE:
                print "ILLEGAL MOVE!"
            self.game_over = True
            return -1

        rotation, position = self.actionToRotPos(action)

        if VERBOSE:
            print "Playing piece:\n", action, rotation, position

        self.removePieceFromBoard()
        self.current_piece = self.current_piece.rotate(rotation)
        self.current_piece.setColumn(position)

        if VISUALIZE:
            self.addPieceToBoard()
            self.visualizeBoard("screenshots/frame%08d.png" % self.frame_count)
            self.frame_count += 1
            self.removePieceFromBoard()

        self.game_over = not self.dropCurrentPiece()

        if VISUALIZE:
            self.visualizeBoard("screenshots/frame%08d.png" % self.frame_count)
            self.frame_count += 1

        if self.game_over:
            if VERBOSE:
                print "hit the sky"
            return -1

        rows_cleared = self.clearRows()
        if rows_cleared > 0 and VISUALIZE:
            self.visualizeBoard("screenshots/frame%08d.png" % self.frame_count)
            self.frame_count += 1

        self.current_piece = self.getRandomPiece()
        self.addPieceToBoard()
        if VERBOSE:
            self.printBoard()

        if VISUALIZE:
            self.visualizeBoard("screenshots/frame%08d.png" % self.frame_count)
            self.frame_count += 1

        # self.games_file.write(self.getBoardString())
        # self.games_file.flush()

        # return min(rows_cleared, 1)
        return 1  # yay, survived another move!
        # print "ROWS CLEARED:", rows_cleared
        # return rows_cleared

    def isEpisodeOver(self):
        return self.game_over

    def reset(self):
        # print "RESET"
        #self.current_game_file.close()
        #self.current_game_file = open("current_game.txt", 'w')
        self.game_over = False
        self.board = np.zeros((self.board_height, self.board_width))
        self.current_piece = self.getRandomPiece()
        self.addPieceToBoard()
        self.game_moves = 0
        # self.printBoard()
        # if self.display_animation:
        #     self.board_image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
        #     self.board_drawer = ImageDraw.Draw(self.board_image)

if __name__ == "__main__":
    ste = SimpleTetrisEnvironment()
    for i in range(10):
        ste.performRandomAction()
        ste.printBoard()


