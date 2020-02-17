from . import config

istorage_prepared_st = config.session.prepare('INSERT INTO hecuba.istorage'
                                              '(storage_id, table_name, obj_name, data_model, tokens)'
                                              'VALUES (?,?,?,?,?)')
istorage_insert_tokens = config.session.prepare('INSERT INTO hecuba.istorage'
                                                '(storage_id, tokens)'
                                                'VALUES (?,?)')
istorage_remove_entry = config.session.prepare('DELETE FROM hecuba.istorage WHERE storage_id = ?')
istorage_read_entry = config.session.prepare('SELECT * FROM hecuba.istorage WHERE storage_id = ?')
