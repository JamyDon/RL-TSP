'''
    __main__: the main python file
'''

import q_learning
import policy_gradient

def main():
    q_learning.q_learning()
    policy_gradient.policy_gradient()

if __name__ == '__main__':
    main()