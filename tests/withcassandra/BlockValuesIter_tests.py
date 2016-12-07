import unittest

from hecuba import config, Config


class BlockValuesIterTest(unittest.TestCase):
    @staticmethod
    def setUpClass():
        Config.reset(mock_cassandra=False)

    def test_init(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp1')
        config.session.execute(
            "CREATE KEYSPACE ksp1 WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp1.tt1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        partition_key = 'pk1'
        keyspace = 'ksp1'
        table = 'tt1'
        self._query = config.session.prepare(
            "SELECT * FROM " + keyspace + "." + table + " WHERE " +
            "token(" + partition_key + ") >= ? AND " +
            "token(" + partition_key + ") < ?")
        metadata = config.cluster.metadata
        ringtokens = metadata.token_map
        block_tokens = [8508619251581300691, 8514581128764531689, 8577968535836399533, 8596162846302799189,
                        8603491526474728284, 8628291680139169981, 8687301163739303017, 9111581078517061776]
        block_tokens2 = [-9094437162685530761, -8915528750990858804, -8794897976085586879, -8789705479149566761,
                         -8706660379068013694, -8679327346684584463, -8654067310550193078, -8636353559349213621]

        ran = set(block_tokens)
        ran2 = set(block_tokens2)
        last = ringtokens.ring[len(ringtokens.ring) - 1]
        self._token_ranges = []
        self._token_ranges2 = []
        max_token = -9223372036854775808
        min_token = 9223372036854775807

        for t in ringtokens.ring:
            if t.value > max_token:
                max_token = t.value
            if t.value < min_token:
                min_token = t.value

        for t in ringtokens.ring:
            if t.value in ran:
                if t.value == min_token:
                    self._token_ranges.append((-9223372036854775808, min_token))
                    self._token_ranges.append((max_token, 9223372036854775807))
                else:
                    self._token_ranges.append((last, t.value))
            if t.value in ran2:
                if t.value == min_token:
                    self._token_ranges2.append((-9223372036854775808, min_token))
                    self._token_ranges2.append((max_token, 9223372036854775807))
                else:
                    self._token_ranges2.append((last, t.value))
            last = t.value

        self.assertEqual(len(block_tokens), len(self._token_ranges))
        self.assertEqual(len(block_tokens) + 1, len(self._token_ranges2))
        for entry in self._token_ranges2:
            if entry[0] == -9223372036854775808:
                self.assertEqual(entry[1], min(block_tokens2))
            if entry[1] == 9223372036854775807:
                self.assertEqual(entry[0], max_token)

    def test_next(self):
        config.session.execute('DROP KEYSPACE IF EXISTS ksp1')
        config.session.execute("CREATE KEYSPACE ksp1 WITH replication = " +
                               "{'class': 'SimpleStrategy', 'replication_factor': 1};")
        config.session.execute("CREATE TABLE ksp1.tt1(pk1 int, val1 text,PRIMARY KEY(pk1))")
        config.session.execute("INSERT INTO ksp1.tt1(pk1, val1) VALUES (1, 'BLABLABLA')")
        partition_key = 'pk1'
        keyspace = 'ksp1'
        table = 'tt1'
        self._query = config.session.prepare(
            "SELECT * FROM " + keyspace + "." + table + " WHERE " +
            "token(" + partition_key + ") >= ? AND " +
            "token(" + partition_key + ") < ?")
        self._token_ranges = [(-9223372036854775808, 9223372036854775807)]
        self._token_pos = 0
        self._current_iterator = None

        if self._token_pos < len(self._token_ranges):
            query = self._query.bind(self._token_ranges[self._token_pos])
            self._current_iterator = iter(config.session.execute(query))
            self.assertEquals(self._current_iterator.current_rows[0].pk1, 1)
            self.assertEquals(self._current_iterator.current_rows[0].val1, 'BLABLABLA')
