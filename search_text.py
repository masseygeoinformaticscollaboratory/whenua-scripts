from pymongo import MongoClient


class Command:

    def run(self, keyword):
        client = MongoClient('mongodb://admin:6677028xxbbkat@hpc-mongodb01.massey.ac.nz:27017/whenua')
        mydb = client.whenua
        mycol = mydb.AllData

        docs = mycol.find(
            {
                '$text':
                    {
                        '$search': keyword,
                        '$caseSensitive': True,
                        '$diacriticSensitive': True
                    }
            }
        )

        print('Found {} records with keyword {}.'.format(docs.count(), keyword))
        print('Here are the first 3 records')
        for doc in docs[:3]:
            print('----------------------------------------')
            print(doc)
            print('----------------------------------------')


if __name__ == '__main__':
    keyword = 'whenua'
    command = Command()
    command.run(keyword)
