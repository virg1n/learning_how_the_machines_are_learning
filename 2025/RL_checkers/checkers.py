import numpy as np

class Checkers:
    def __init__(self):
        self.black = np.zeros((10, 10))
        self.white = np.zeros((10, 10))
        for row in range(1, 4):
            for col in range(row % 2 + 1, 9, 2):
                self.black[row][col] = 1

        for row in range(6, 9):
            for col in range(row % 2 + 1, 9, 2):
                self.white[row][col] = 1
        
        board = np.where(self.black, "b", "_") 
        self.board = np.where(self.white, "w", board)
        self.turn = "w"

    def show(self):
        print(self.board)

    def GetPossibleMovesForPawn(self, x, y, turn=None):
        if turn is None:
            turn = self.turn

        if self.board[y][x] != turn:
            return []

        if turn == "w":
            symb = -1
        else:
            symb = 1
        
        moves = self.CanEatForPawn(x, y, turn)
        if not moves:
            if (y == 1 and turn == "w") or (y == 8 and turn == "b"):
                return None  # todo: implement king moves

            if InRange(x+1, y+symb) and self.board[y+symb][x+1] == "_":
                moves.append([x+1, y+symb])

            if InRange(x-1, y+symb) and self.board[y+symb][x-1] == "_":
                moves.append([x-1, y+symb])

        return moves

    def CanEatForPawn(self, x, y, turn=None):
        if turn is None:
            turn = self.turn
        
        other_turn = "b" if turn == "w" else "w"
        moves = []

        # top-left
        if InRange(x-1, y-1) and self.board[y-1][x-1] == other_turn:
            if InRange(x-2, y-2) and self.board[y-2][x-2] == "_":
                # REMOVE pawn from (x,y):
                self.board[y][x] = "_"
                # REMOVE the jumped piece:
                self.board[y-1][x-1] = "_"
                # PLACE pawn at landing square:
                self.board[y-2][x-2] = turn

                moves.append([x-2, y-2])
                moves.extend(self.CanEatForPawn(x-2, y-2, turn))

                # RESTORE landing square:
                self.board[y-2][x-2] = "_"
                # RESTORE jumped piece:
                self.board[y-1][x-1] = other_turn
                # RESTORE pawn to original square:
                self.board[y][x] = turn

        # top-right
        if InRange(x+1, y-1) and self.board[y-1][x+1] == other_turn:
            if InRange(x+2, y-2) and self.board[y-2][x+2] == "_":
                self.board[y][x] = "_"
                self.board[y-1][x+1] = "_"
                self.board[y-2][x+2] = turn

                moves.append([x+2, y-2])
                moves.extend(self.CanEatForPawn(x+2, y-2, turn))

                self.board[y-2][x+2] = "_"
                self.board[y-1][x+1] = other_turn
                self.board[y][x] = turn

        # bottom-left
        if InRange(x-1, y+1) and self.board[y+1][x-1] == other_turn:
            if InRange(x-2, y+2) and self.board[y+2][x-2] == "_":
                self.board[y][x] = "_"
                self.board[y+1][x-1] = "_"
                self.board[y+2][x-2] = turn

                moves.append([x-2, y+2])
                moves.extend(self.CanEatForPawn(x-2, y+2, turn))

                self.board[y+2][x-2] = "_"
                self.board[y+1][x-1] = other_turn
                self.board[y][x] = turn

        # bottom-right
        if InRange(x+1, y+1) and self.board[y+1][x+1] == other_turn:
            if InRange(x+2, y+2) and self.board[y+2][x+2] == "_":
                self.board[y][x] = "_"
                self.board[y+1][x+1] = "_"
                self.board[y+2][x+2] = turn

                moves.append([x+2, y+2])
                moves.extend(self.CanEatForPawn(x+2, y+2, turn))

                self.board[y+2][x+2] = "_"
                self.board[y+1][x+1] = other_turn
                self.board[y][x] = turn

        return moves

        
        
    def GetPossibleMove(self, turn=None):
        if turn is None:
            turn = self.turn
        
        


def InRange(x, y):
    return 1 <= x <= 8 and 1 <= y <= 8


def main():
    check = Checkers()
    check.show()

if __name__ == "__main__":
    main()


# 0, 2, 4, 6
# 1 0 1 0 1 0 1 0
# 0 1 0 1 0 1 0 1
# 1 0 1 0 1 0 1 0
# 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0
# 0 1 0 1 0 1 0 1
# 1 0 1 0 1 0 1 0
# 0 1 0 1 0 1 0 1


# black = [1 0 1 0 1 0 1 0, 
# 0 1 0 1 0 1 0 1,
# 1 0 1 0 1 0 1 0,
# 0, 0, 0..., 0, 0]



