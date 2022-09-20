import logging

from gensim.utils import check_output

from ldamallet import LdaMallet
from gensim import utils


logger = logging.getLogger(__name__)


class HLdaMallet (LdaMallet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def corpus_to_mallet_input(self, corpus, mallet_input_file):
        with utils.open(mallet_input_file, 'wb') as fout:
            self.corpus2mallet(corpus, fout)

    def input_to_mallet(self, mallet_input_file, mallet_output_file):
        # convert the text file above into MALLET's internal format
        cmd = \
            self.mallet_path + " import-file --keep-sequence --remove-stopwords --input %s --output %s"
        cmd = cmd % (mallet_input_file, mallet_output_file)
        logger.info("converting temporary corpus to MALLET format with %s", cmd)
        check_output(args=cmd, shell=True)

    def get_train_cmd(self, mallet_file, output_file):
        cmd = self.mallet_path + " run cc.mallet.topics.tui.HierarchicalLDATUI --input %s --output-state %s"
        cmd = cmd % (mallet_file, output_file)
        return cmd

    def train(self, mallet_file, output_file):
        cmd = self.get_train_cmd(mallet_file, output_file)
        logger.info("training MALLET HLDA with %s", cmd)
        check_output(args=cmd, shell=True)
