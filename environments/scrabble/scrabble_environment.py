import string
import numpy as np
import random
import time
import sys
import cv2
from environment import Environment

VERBOSE = False
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
VISUAL_DELAY = 0.1
VISUALIZE = False


class ScrabbleBoard:
    
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype="S1")
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                self.board[row][col] = " "
                
    def addTile(self, row, col, letter):
        self.board[row][col] = letter
        
    def getTile(self, row, col):
        return self.board[row][col]
        
    def isEmpty(self):
        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.getTile(row, col) != ' ':
                    return False
        return True
        
    def __str__(self):
        string_width = 4 * self.board_size + 1
        result = ''
        result += "-" * string_width
        result += '\n'

        for row in range(self.board.shape[0]):
            result += "| "
            for col in range(self.board.shape[1]):
                result += self.board[row][col] + " | "
            result += '\n'

            result += "-" * string_width
            result += '\n'
        return result


class ScrabbleGame:

    tile_distribution = "eeeeeeeeeeeeaaaaaaaaaiiiiiiiiioooooooonnnnnnrrrrrrttttttllllssssuuuuddddgggbbccmmppffhhvvwwyykjxqz"

    def __init__(self, board_size=7):
        self.reset()
        self.word_list = {}
        word_list_file = open("short_words.txt", 'r')
        for line in word_list_file:
            self.word_list[line.strip()] = True
        
    def getActionSet(self):
        return range(10)  # only 10 actions??????
        
    def hasTiles(self, move):
        rack_copy = self.rack[:]
        for m in move.values():
            if m not in rack_copy:
                return False
            else:
                rack_copy.remove(m)

        return True
    
    
    def getScreen(self):
        # Build ANN input board
        # 26 = 1-hot encoding for letters
        new_board = np.zeros((board.board_size, board.board_size * 26))

        for row_number in range(board.board_size):
            for col_number in range(board.board_size):  # go through the original board values
                curr_letter = board.getTile(row_number, col_number)
                if curr_letter != ' ':
                    letter_index = string.lowercase.index(curr_letter)
                    print curr_letter, letter_index
                    new_board[row_number][col_number * 26 + letter_index] = 1.0
                    
        return new_board
    
    
    def performRandomAction(self):
        move = {
            (2,5):'t',
            (3,5):'o',
            #(4,5):'G',
            }
    
        self.performMove(move)

    
    def performAction(self, action):
        
        # action is a single int...
        

        move = {
            (2,5):'t',
            (3,5):'o',
            #(4,5):'G',
            }
               
        reward = 0
        move = p.makeMove(self.board)
        if p.hasTiles(move):
            if self.isLegalMove(move):
                self.performMove(move)
                reward = len(move) + 1
            else:
                print "ILLEGAL MOVE:", move
                reward = -1
        else:
            print "DON't have the tiles!"
            reward = -1

       
        # call performMove after converting action number to a move dictionary
        self.performMove(move)
       
  
        return 0 # return the reward for the play
      
    
    def pickRandomTile(self):
        return self.tile_bag.pop(random.randrange(len(self.tile_bag)))
    
    def drawInitialTiles(self, rack_size=7):
        for i in range(rack_size):
            self.rack += self.pickRandomTile()
        
    def initTiles(self, number_tiles = 100):
        tiles = []
        for i in range(number_tiles):
            tiles.append(random.choice(self.tile_distribution))
        return tiles
    
    def performMove(self, move):
        for key, val in move.items():
            row, col = key
            self.board.addTile(row, col, val)
    
    
    def isMoveWithinBounds(self, move):
        for (row, col), tile in move.items():
            if row < 0 or row >= self.board.board_size or col < 0 or col >= self.board.board_size:
                return False
        return True

    def isMoveOverlapping(self, move):
        for (row, col), tile in move.items():
            if self.board.getTile(row, col) != " ":
                return True
        return False
    
    def isMoveStraight(self, move):
        rows = []
        cols = []
        for (row, col), tile in move.items():
            rows.append(row)
            cols.append(col)
            
        same_rows = True
        first_row = rows[0]
        for row in rows[1:]:
            if row != first_row:
                same_rows = False
                
        same_cols = True
        first_col = cols[0]
        for col in cols[1:]:
            if col != first_col:
                same_cols = False

        return same_rows or same_cols

    def isMoveHorizontal(self, move):
        move_keys = move.keys()
        first_col = move_keys[0][1]
        for (row, col) in move_keys[1:]:
            if col == first_col:
                return False
        return True
            
    def undoMove(self, move):
        for key, val in move.items():
            row, col = key
            self.board.addTile(row, col, ' ')

    def isMoveGapless(self, move):
        # fake perform move
        self.performMove(move)
        rows = []
        cols = []
        for (row, col), tile in move.items():
            rows.append(row)
            cols.append(col)
            
        #print rows, cols
            
        if self.isMoveHorizontal(move):
            curr_row = rows[0]
            min_col, max_col = min(cols), max(cols)
            print min_col, max_col
        
            for i in range(min_col, max_col + 1):
                print i
                if self.board.getTile(curr_row, i) == ' ':
                    print "FOUND horizontal gap"
                    self.undoMove(move)
                    return False
        else:
            curr_col = cols[0]
            min_row, max_row = min(rows), max(rows)
        
            for i in range(min_row, max_row + 1):
                if self.board.getTile(i, curr_col) == ' ':
                    print "FOUND vertical gap"
                    self.undoMove(move)
                    return False
                
        self.undoMove(move)
        return True
                
    def hasNeighborTile(self, row, col):
        if row > 0 and self.board.getTile(row-1, col) != ' ':
            return True
        elif row < self.board.board_size - 1 and self.board.getTile(row+1, col) != ' ':
            return True
        elif col > 0 and self.board.getTile(row, col-1) != ' ':
            return True
        elif col < self.board.board_size - 1 and self.board.getTile(row, col+1) != ' ':
            return True
        return False
                
    def isMoveAdjacentToExisting(self, move):
        # First move
        if self.board.isEmpty():
            return True
        
        for (row, col), tile in move.items():
            if self.hasNeighborTile(row, col):
                return True
        return False
    
    
    def getVerticalWord(self, row, col):
        curr_row = row
        word = self.board.getTile(curr_row, col)
        curr_row += 1
        while curr_row < self.board.board_size and self.board.getTile(curr_row, col) != ' ':
            word += self.board.getTile(curr_row, col)
            curr_row += 1
        return word
    
    def getHorizontalWord(self, row, col):
        curr_col = col
        word = self.board.getTile(row, curr_col)
        curr_col += 1
        while curr_col < self.board.board_size and self.board.getTile(row, curr_col) != ' ':
            word += self.board.getTile(row, curr_col)
            curr_col += 1
        return word

    def isLegalWord(self, word):
        print "Checking word:", word
        if len(word) < 2:
            return True
        else:
            return word in self.word_list
    
    def boardWouldBeLegal(self, move):
        
        # Fake move
        self.performMove(move)
        
        # check the verticals
        for col in range(self.board.board_size):
            row = 0
            while row < self.board.board_size:
                if self.board.getTile(row, col) == ' ':
                    row += 1
                else:
                    word = self.getVerticalWord(row, col)
                    if not self.isLegalWord(word):
                        self.undoMove(move)
                        return False
                    else:
                        row += len(word) + 1
        
        # check the horizontals
        for row in range(self.board.board_size):
            col = 0
            while col < self.board.board_size:
                if self.board.getTile(row, col) == ' ':
                    col += 1
                else:
                    word = self.getHorizontalWord(row, col)
                    if not self.isLegalWord(word):
                        self.undoMove(move)
                        return False
                    else:
                        col += len(word) + 1
        
        self.undoMove(move)
        return True

    def isLegalMove(self, move):
        
        # is within the bounds of the board
        if not self.isMoveWithinBounds(move):
            print "ILLEGAL: MOVE OUT OF BOUNDS"
            return False
        
        # does not overlap with existing tiles
        if self.isMoveOverlapping(move):
            print "ILLEGAL: MOVE OVERLAPPING"
            return False
        
        # is in a straight line
        if not self.isMoveStraight(move):
            print "ILLEGAL: MOVE NOT STRAIGHT LINE"
            return False
        
        # is without gaps along its straight line
        if not self.isMoveGapless(move):
            print "ILLEGAL: MOVE IS NOT GAPLESS"
            return False

        # is adjacent to an existing word (or is first word in the middle)
        if not self.isMoveAdjacentToExisting(move):
            print "ILLEGAL: MOVE IS NOT ADJACENT TO OTHER TILES"
            return False
        
        # every word formed is a word from the word list
        if not self.boardWouldBeLegal(move):
            print "ILLEGAL: NOT ALL FORMED WORDS ARE IN WORD LIST"
            return False
        
        # Everything passed
        return True
                  
              
    def reset(self):
        self.game_over = False
        self.tile_bag = self.initTiles()
        self.rack = []
        self.board = ScrabbleBoard(board_size)
    
    
    def isEpisodeOver(self):
        return self.game_over
    
              
    def __str__(self):
        result = ''
        result += '\n'
        result += str(self.board)
        
        result += '\n'
        
        print self.rack
            
        return result



