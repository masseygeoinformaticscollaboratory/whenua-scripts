import abc
import argparse
import json
import time

from AbstractCommand import AbstractCommand


class Serialisable:

    @abc.abstractmethod
    def to_json(self):
        pass


class DataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Serialisable):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


class Tree(Serialisable):

    def __init__(self, level, branch_no, parent):
        self.level = level
        self.branches = {}
        self.word_ids = set()
        self.up_branch = parent
        self.branch_no = branch_no
        self.word_by_id = None

    def add_word_id(self, word_id):
        self.word_ids.add(word_id)
        if self.up_branch is not None:
            self.up_branch.add_word_id(word_id)

    def get_branch(self, branch_no):
        branch = self.branches.get(branch_no, None)
        if branch is None:
            branch = Tree(self.level + 1, branch_no, self)
            self.branches[branch_no] = branch
        return branch

    def __str__(self):
        return '{}: {} branches & {} words'.format(self.branch_no, len(self.branches), len(self.word_ids))

    def __unicode__(self):
        return self.__str__()

    def set_word_by_id(self, word_by_id):
        self.word_by_id = word_by_id
        for branch in self.branches.values():
            branch.set_word_by_id(self.word_by_id)

    def get_words(self):
        return [self.word_by_id[word_id] for word_id in self.word_ids]

    def to_json(self):
        if len(self.branches) == 0:
            return dict(name=str(self.branch_no), words=self.get_words())
        else:
            return dict(name=str(self.branch_no), words=self.get_words(), children=list(self.branches.values()))


class Command(AbstractCommand):

    def __init__(self):
        super(Command, self).__init__(__file__)

    def make_tree(self, source):
        self.word_by_id = {}
        self.id_by_word = {}
        self.root = Tree(0, 0, None)
        with open(source, 'r') as f:
            for line in f:
                line = line.strip()
                components = line.split(' ')
                branch_nos = list(map(int, components[:-4]))
                parent = self.root
                for branch_level, branch_no in enumerate(branch_nos[::-1]):
                    branch = parent.get_branch(branch_no)
                    word = components[-2]
                    word_id = int(components[-3])
                    self.word_by_id[word_id] = word
                    self.id_by_word[word] = word_id
                    branch.add_word_id(word_id)
                    parent = branch

    def print_tree(self, tree):
        padding = '__' * tree.level
        print('{}{}'.format(padding, tree.branch_no), end='')
        if len(tree.branches) == 0:
            print(':[{}]'.format(', '.join(tree.get_words(self.word_by_id))))
        else:
            print(':')
        for branch in tree.branches.values():
            self.print_tree(branch)

    def run(self, args):
        source = args.source
        dest = args.dest
        self.make_tree(source)
        # self.print_tree(self.root)
        self.root.set_word_by_id(self.word_by_id)
        tree = dict(name="Root", children=list(self.root.branches.values()), words=['Root'])
        with open(dest, 'w') as f:
            json.dump(tree, f, cls=DataEncoder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', dest='source', required=True, type=str)
    parser.add_argument('--dest', action='store', dest='dest', required=True, type=str)
    args = parser.parse_args()

    command = Command()
    start = time.time()
    command.run(args)
    end = time.time()
    print('Took {} seconds'.format(end - start))


if __name__ == '__main__':
    main()
