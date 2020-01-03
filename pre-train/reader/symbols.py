phone_list = ['pau', 'iy', 'aa', 'ch', 'ae', 'eh', 
 'ah', 'ao', 'ih', 'ey', 'aw', 
 'ay', 'ax', 'er', 'ng', 
 'sh', 'th', 'uh', 'zh', 'oy', 
 'dh', 'y', 'hh', 'jh', 'b', 
 'd', 'g', 'f', 'k', 'm', 
 'l', 'n', 'p', 's', 'r', 
 't', 'w', 'v', 'ow', 'z', 
 'uw', 'SOS/EOS']

seen_speakers = ['p226', 'p230', 'p234', 'p239', 'p244', 'p248', 'p252', 'p256', 'p260', 'p264', 'p268', 'p272', 'p276', 'p280', 'p284', 'p288', 'p295', 'p300', 'p304', 'p308', 'p313', 'p318', 'p330', 'p336', 'p343', 'p360', 'p364', 'p227', 'p231', 'p236', 'p240', 'p245', 'p249', 'p253', 'p257', 'p261', 'p265', 'p269', 'p273', 'p277', 'p281', 'p285', 'p292', 'p297', 'p301', 'p305', 'p310', 'p314', 'p323', 'p333', 'p339', 'p345', 'p361', 'p374', 'p228', 'p232', 'p237', 'p241', 'p246', 'p250', 'p254', 'p258', 'p262', 'p266', 'p270', 'p274', 'p278', 'p282', 'p286', 'p293', 'p298', 'p302', 'p306', 'p311', 'p316', 'p326', 'p334', 'p340', 'p347', 'p362', 'p376', 'p225', 'p229', 'p233', 'p238', 'p243', 'p247', 'p251', 'p255', 'p259', 'p263', 'p267', 'p271', 'p275', 'p279', 'p283', 'p287', 'p294', 'p299', 'p303', 'p307', 'p312', 'p317', 'p329', 'p335', 'p341', 'p351', 'p363']

ph2id = {ph:i for i, ph in enumerate(phone_list)}
id2ph = {i:ph for i, ph in enumerate(phone_list)}
sp2id = {sp:i for i, sp in enumerate(seen_speakers)}
id2sp = {i:sp for i, sp in enumerate(seen_speakers)}
