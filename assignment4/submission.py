import sys
import math
from collections import namedtuple, defaultdict
from itertools import chain, product

START_SYM = 'ROOT'


class GrammarRule(namedtuple('Rule', ['lhs', 'rhs', 'log_probability'])):
    """A named tuple that represents a PCFG grammar rule.

    Each GrammarRule has three fields: lhs, rhs, log_probability

    Parameters
    ----------
    lhs : str
        A string that represents the left-hand-side symbol of the grammar rule.
    rhs : tuple of str
        A tuple that represents the right-hand-side symbols the grammar rule.
    log_probability : float
        The log probability of this rule.
    """
    def __repr__(self):
        return '{} -> {} [{}]'.format(
            self.lhs, ' '.join(self.rhs), self.log_probability)


def read_rules(grammar_filename):
    """Read PCFG grammar rules from grammar file

    The grammar file is a tab-separated file of three columns:
    probability, left-hand-side, right-hand-side.
    probability is a float number between 0 and 1. left-hand-side is a
    string token for a non-terminal symbol in the PCFG. right-hand-side
    is a space-delimited field for one or more  terminal and non-terminal
    tokens. For example::

        1	ROOT	EXPR
        0.333333	EXPR	EXPR + TERM

    Parameters
    ----------
    grammar_filename : str
        path to PCFG grammar file

    Returns
    -------
    set of GrammarRule
    """
    rules = set()
    with open(grammar_filename) as f:
        for rule in f.readlines():
            rule = rule.strip()
            log_probability, lhs, rhs = rule.split('\t')
            rhs = tuple(rhs.split(' '))
            assert rhs and rhs[0], rule
            rules.add(GrammarRule(lhs, rhs, math.log(float(log_probability))))
    return rules


class Grammar:
    """PCFG Grammar class."""
    def __init__(self, rules):
        """Construct a Grammar object from a set of rules.

        Parameters
        ----------
        rules : set of GrammarRule
            The set of grammar rules of this PCFG.
        """
        self.rules = rules

        self._rhs_rules = defaultdict(list)
        self._rhs_unary_rules = defaultdict(list)

        self._nonterm = set(rule.lhs for rule in rules)
        self._term = set(token for rhs in chain(rule.rhs for rule in rules)
                         for token in rhs if token not in self._nonterm)

        for rule in rules:
            _, rhs, _ = rule
            self._rhs_rules[rhs].append(rule)

        for rhs_rules in self._rhs_rules.values():
            rhs_rules.sort(key=lambda r: r.log_probability, reverse=True)

        self._is_cnf = all(len(rule.rhs) == 1
                           or (len(rule.rhs) == 2
                               and all(s in self._nonterm for s in rule.rhs))
                           for rule in self.rules)

    def from_rhs(self, rhs):
        """Look up rules that produce rhs

        Parameters
        ----------
        rhs : tuple of str
            The tuple that represents the rhs.

        Returns
        -------
        list of GrammarRules with matching rhs, ordered by their
        log probabilities in decreasing order.
        """
        return self._rhs_rules[rhs]

    def __repr__(self):
        summary = 'Grammar(Rules: {}, Term: {}, Non-term: {})\n'.format(
            len(self.rules), len(self.terminal), len(self.nonterminal)
        )
        summary += '\n'.join(sorted(self.rules))
        return summary

    @property
    def terminal(self):
        """Terminal tokens in this grammar."""
        return self._term

    @property
    def nonterminal(self):
        """Non-terminal tokens in this grammar."""
        return self._nonterm

    def get_cnf(self):
        """Convert PCFG to CNF and return it as a new grammar object."""
        nonterm = set(self.nonterminal)
        term = set(self.terminal)

        rules = list(self.rules)
        cnf = set()

        # STEP 1: eliminate nonsolitary terminals
        for i in range(len(rules)):
            rule = rules[i]
            lhs, rhs, log_probability = rule
            if len(rhs) > 1:
                rhs_list = list(rhs)
                for j in range(len(rhs_list)):
                    x = rhs_list[j]
                    if x in term:  # found nonsolitary terminal
                        new_nonterm = 'NT_{}'.format(x)
                        new_nonterm_rule = GrammarRule(new_nonterm, (x,), 0.0)

                        if new_nonterm not in nonterm:
                            nonterm.add(new_nonterm)
                            cnf.add(new_nonterm_rule)
                        else:
                            assert new_nonterm_rule in cnf
                        rhs_list[j] = new_nonterm
                rhs = tuple(rhs_list)
            rules[i] = GrammarRule(lhs, rhs, log_probability)

        # STEP 2: eliminate rhs with more than 2 nonterminals
        for i in range(len(rules)):
            rule = rules[i]
            lhs, rhs, log_probability = rule
            if len(rhs) > 2:
                assert all(x in nonterm for x in rhs), rule
                current_lhs = lhs
                for j in range(len(rhs) - 2):
                    new_nonterm = 'BIN_"{}"_{}'.format(
                        '{}->{}'.format(lhs, ','.join(rhs)), str(j))
                    assert new_nonterm not in nonterm, rule
                    nonterm.add(new_nonterm)
                    cnf.add(
                        GrammarRule(current_lhs,
                                    (rhs[j], new_nonterm),
                                    log_probability if j == 0 else 0.0))
                    current_lhs = new_nonterm
                cnf.add(GrammarRule(current_lhs, (rhs[-2], rhs[-1]), 0.0))
            else:
                cnf.add(rule)

        return Grammar(cnf)

    def parse(self, line):
        """Parse a sentence with the current grammar.

        The grammar object must be in the Chomsky normal form.

        Parameters
        ----------
        line : str
            Space-delimited tokens of a sentence.
        """
        # BEGIN_YOUR_CODE
        def print_tree(tree,curr,space):
            if(curr > len(tree)-1):
                return ""
            elif(len(tree[curr][1])>1):
                k = curr
                count = 0
                for i in range(0,len(tree)):
                    if(len(tree[i][1])>1 and (tree[i][1][1] == tree[curr][1][1])):
                        count += 1
                        if(len(tree[i][1])>1 and (tree[i][1][1] == tree[curr][1][1] and tree[i][1][0] == tree[curr][1][1])):
                            count += 1
                    if(i>0 and tree[i][0]==tree[curr][1][1]):
                        k=i
                        count -= 1
                        if(len(tree[i][1])>1 and tree[i][1][1] == tree[curr][1][1]):
                            count -= 1
                        if(count <= 0):
                            break
                lhs = tree[curr+1:k]
                rhs = tree[k:len(tree)]
                spaces = " " * space
                if("BIN_" in tree[curr][0]):
                    str = '{}'.format(print_tree(lhs,0,space))
                    str += '\n{}'.format(spaces)
                    str += '{}'.format(print_tree(rhs,0,space))
                    return str
                else:
                    str = '({} '.format(tree[curr][0])
                    space += len(tree[curr][0]) + 2
                    str += '{}'.format(print_tree(lhs,0,space))
                    spaces = " " * space
                    str += '\n{}'.format(spaces)
                    str += '{})'.format(print_tree(rhs,0,space))
                    return str
            else:
                space += len(tree[curr][0]) + 2
                if(curr < len(tree) - 1 and tree[curr][1][0] == tree[curr+1][0]):
                    lhs = tree[curr+1:len(tree)]
                    if("NT_" in tree[curr][0]):
                        str = '{}'.format(print_tree(lhs,0,space))
                        return str
                    else:
                        str = '({} '.format(tree[curr][0])
                        str += '{})'.format(print_tree(lhs,0,space))
                        return str
                else:
                    if("NT_" in tree[curr][0]):
                        str = '{}'.format(tree[curr][1][0])
                        return str
                    else:
                        str = '({} {})'.format(tree[curr][0],tree[curr][1][0])
                        return str
        # *************************************************************************************************************
        
        input = line.split()
        n = len(input)
        bestScore = {}
        backPointer = {}
        binary_rules = []
        unary_rules = []
        for rule in self.rules:
            A, rhs, log_probability = rule
            if(len(rhs) == 1):
                unary_rules.append(rule)
            else:
                binary_rules.append(rule)
        
        # initialize bestScore and backPointer
        for i in range(0, n):
            for j in range(i + 1, n + 1):
                currScore = {}
                currPointer = {}
                for rule in self.rules:
                    A, rhs, log_probability = rule
                    currScore[A] = 0.0
                    rule_list = []
                    currPointer[A] = (i, rule_list)
                bestScore[(i,j)] = currScore 
                backPointer[(i,j)] = currPointer
        
        # fill in terminal rules
        for i in range(1, n + 1):
            for rule in unary_rules:
                A, rhs, log_probability = rule
                p = math.exp(log_probability)
                if(input[i-1] in rhs and p > bestScore[(i-1, i)][A]):
                    bestScore[(i-1,i)][A] = p
                    best_rules = [rule]
                    if(len(backPointer[(i-1, i)][A][1]) > 0):
                        best_rules += backPointer[(i-1, i)][A][1]
                    backPointer[(i-1, i)][A] = (i-1, best_rules)
            
            found = True
            unseen = []
            unseen += unary_rules
            LHS = []
            for d in bestScore[(i-1, i)]:
                if(bestScore[(i-1, i)][d]>0.0):
                    LHS.append(d)
            while(found):
                found = False
                for A in LHS:
                    for rule in unseen:
                        B, rhs, log_probability = rule
                        p = math.exp(log_probability)
                        if(A in rhs):
                            score = p * bestScore[(i-1, i)][A]
                            if(score > bestScore[(i-1, i)][B]):
                                bestScore[(i-1, i)][B] = score
                                best_rules = [rule]
                                if(len(backPointer[(i-1, i)][A][1])>0):
                                    best_rules += backPointer[(i-1,i)][A][1]
                                if(len(backPointer[(i-1, i)][B][1])>0):
                                    best_rules += backPointer[(i-1, i)][B][1]
                                backPointer[(i-1, i)][B] = (i-1, best_rules)
                                found = True
                                unseen.remove(rule)
                                LHS.append(B)
                                break
                    if(found):
                        break

        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                j = i + l
                for k in range(i + 1, j):
                    for rule in binary_rules:
                        A, rhs, log_probability = rule
                        p = math.exp(log_probability)
                        B = rhs[0]
                        C = rhs[1]
                        if(bestScore[(i,k)][B] > 0.0 and bestScore[(k,j)][C] > 0.0):
                            score = p * bestScore[(i,k)][B] * bestScore[(k,j)][C]
                            if(score > bestScore[(i,j)][A]):
                                bestScore[(i,j)][A] = score
                                backPointer[(i,j)][A] = (k,[rule] + backPointer[(i,k)][B][1] + backPointer[(k,j)][C][1])
                
                # add unary rules to non-terminals
                found = True
                unseen = []
                unseen += unary_rules
                LHS = []
                for d in bestScore[(i,j)]:
                    if(bestScore[(i,j)][d]>0.0):
                        LHS.append(d)
                while(found):
                    found = False
                    for A in LHS:
                        for rule in unseen:
                            B, rhs, log_probability = rule
                            if(A in rhs):
                                q = math.exp(log_probability)
                                score = q * bestScore[(i,j)][A]
                                if(score > bestScore[(i,j)][B]):
                                    bestScore[(i,j)][B] = score
                                    backPointer[(i,j)][B] = (i,[rule] + backPointer[(i,j)][A][1])
                                    found = True
                                    unseen.remove(rule)
                                    LHS.append(B)
                                    break
                        if(found):
                            break
        
        if(n == 0):
            found = False
        elif(bestScore[(0,n)]['ROOT'] == 0.0):
            print("NONE")
        else:
            tree = backPointer[(0,n)]['ROOT'][1]
            print(print_tree(tree,0,0))
            print(math.log(bestScore[(0,n)]['ROOT']))
        # END_YOUR_CODE