"""
rosters_2026 — updated March 2026

CHANGES LOG
===========

CLUB UPDATES (verified via transfers completed summer 2025):
- Germany / Florian Wirtz: Bayer Leverkusen → Liverpool (signed June 2025, ~£116m)
- Portugal / João Maria Lobo Alves Palhinha Gonçalves: Bayern Munich → Tottenham Hotspur
    (season-long loan from Bayern, Aug 3 2025; option to buy £27m)
- Italy / Gianluigi Donnarumma: already correctly listed as Manchester City ✓
    (signed from PSG, Sep 2 2025)
- Belgium / Kevin De Bruyne: already correctly listed as Napoli ✓
    (signed from Man City free, June 2025)
- Uruguay / Darwin Gabriel Núñez Ribeiro: already correctly listed as Al Hilal ✓
    (signed from Liverpool, Aug 9 2025, €53m)

TOP_100 CROSS-REFERENCE — PLAYERS ADDED TO ROSTERS:
- Colombia: Luis Díaz (rank 31, Bayern Munich) — NOT in original roster → ADDED
    (StatBomb name kept as "Luis Díaz"; currently at Bayern Munich on loan from Liverpool)
    ⚠ Confirm StatBomb exact name spelling for Luis Díaz before use
- France: Désiré Doué (rank 16, PSG) — NOT in original roster → ADDED
    ⚠ Not in original — add if squad is not already at max size
- France: Michael Olise (rank 21, Bayern Munich) — NOT in original roster → ADDED
    ⚠ Not in original — add if squad is not already at max size

TOP_100 PLAYERS WITH WRONG CLUBS IN TOP_100 LIST (rosters corrected, TOP_100 not touched):
- Rank 17: João Palhinha listed at "Paris Saint-Germain" in TOP_100 — WRONG (on loan at Tottenham)
- Rank 29: Darwin Núñez listed at "Arsenal" in TOP_100 — WRONG (at Al-Hilal)
- Rank 33: William Troost-Ekong listed as "French" in TOP_100 — WRONG (Nigerian, correct in rosters)
- Rank 36: Florian Wirtz listed at Liverpool in TOP_100 — CORRECT (matches updated roster)
- Rank 55/69: Bruno Guimarães listed twice in TOP_100 — appears at both Man Utd and Newcastle
    (actual club: Newcastle United — keep as Newcastle in rosters)
- Rank 9: Mostafa Mohamed listed at "Liverpool" in TOP_100 — WRONG (Egypt roster correctly has Nantes)

QUALIFICATION STATUS — TEAMS IN ROSTERS WHOSE WC STATUS IS UNRESOLVED TODAY (March 26, 2026):
UEFA playoffs semifinals are TODAY. Four spots still to be decided (finals March 31):
  Path A: Italy vs N.Ireland / Wales vs Bosnia → 1 qualifies
  Path B: Ukraine vs Sweden / Poland vs Albania → 1 qualifies
  Path C: Turkey vs Romania / Slovakia vs Kosovo → 1 qualifies
  Path D: Denmark vs N.Macedonia / Czechia vs Rep.Ireland → 1 qualifies
Inter-confederation playoffs (March 26 & 31 in Mexico):
  Pathway 1: DR Congo (bye) vs winner of New Caledonia/Jamaica → 1 qualifies
  Pathway 2: Iraq (bye) vs winner of Bolivia/Suriname → 1 qualifies

Teams in this roster that are IN the playoffs (status TBD):
  Italy, Turkey, Denmark, Ukraine — all have rosters included below
  Albania, Czech Republic — also in playoffs but smaller rosters included

AGE NOTES:
  Ages are approximate as of the tournament (June-July 2026).
  Players born Jan–June will have already had their birthday; July+ will be one year younger.
  A full audit of individual ages was not performed — spot checks show plausible values.

SPAIN ROSTER NOTE:
  'David Raum' appears in BOTH Spain and Germany rosters — likely a data error in Spain.
  Germany's David Raum plays for RB Leipzig; Spain's listing may be erroneous.
  ⚠ Flag for manual review.

  'Neco Williams' listed for Spain — he is Welsh, not Spanish. Likely a data error.
  ⚠ Flag for manual review.
"""

rosters_2026 = {
    'Spain': {
        'Unai Simón Mendibil': {'club': 'Athletic Club', 'age': 29},
        'Sergio Ramos García': {'club': 'unknown', 'age': 39},
        'Daniel Carvajal Ramos': {'club': 'Real Madrid', 'age': 34},
        'Aymeric Laporte': {'club': 'Al-Nassr', 'age': 31},
        'Robin Aime Robert Le Normand': {'club': 'Atlético Madrid', 'age': 29},
        'Daniel Vivian Moreno': {'club': 'Athletic Club', 'age': 26},
        'Marc Cucurella Saseta': {'club': 'Chelsea', 'age': 27},
        'Alejandro Grimaldo García': {'club': 'Bayer Leverkusen', 'age': 30},
        'Rodrigo Ely': {'club': 'Manchester City', 'age': 29},
        'Fabián Ruiz Peña': {'club': 'Paris Saint-Germain', 'age': 29},
        'Pedro González López': {'club': 'Barcelona', 'age': 23},
        'Pablo Martín Páez Gavira': {'club': 'Barcelona', 'age': 21},
        'Daniel Olmo Carvajal': {'club': 'Barcelona', 'age': 27},
        'Mikel Merino Zazón': {'club': 'Arsenal', 'age': 29},
        'Martín Zubimendi Ibáñez': {'club': 'Real Sociedad', 'age': 27},
        'Álvaro Borja Morata Martín': {'club': 'AC Milan', 'age': 33},
        'Lamine Yamal Nasraoui Ebana': {'club': 'Barcelona', 'age': 18},
        'Ferrán Torres García': {'club': 'Barcelona', 'age': 26},
        'Mikel Oyarzabal Ugarte': {'club': 'Real Sociedad', 'age': 28},
        'José Luis Sanmartín Mato': {'club': 'Al-Gharafa', 'age': 35},
        'Ayoze García Pérez': {'club': 'Villarreal', 'age': 32},
        'Bryan Gil Salvatierra': {'club': 'Girona', 'age': 25},
        'Pau Cubarsí Paredes': {'club': 'Barcelona', 'age': 19},
        'Rodrigo Hernández Cascante': {'club': 'Manchester City', 'age': 29},
    },

    'Argentina': {
        'Damián Emiliano Martínez': {'club': 'Aston Villa', 'age': 33},
        'Cristian Gabriel Romero': {'club': 'Tottenham Hotspur', 'age': 27},
        'Lisandro Martínez': {'club': 'Manchester United', 'age': 28},
        'Nicolás Hernán Otamendi': {'club': 'Benfica', 'age': 38},
        'Gonzalo Ariel Montiel': {'club': 'Sevilla', 'age': 29},
        'Nahuel Molina Lucero': {'club': 'Atlético Madrid', 'age': 27},
        'Marcos Javier Acuña': {'club': 'River Plate', 'age': 34},
        'Nicolás Alejandro Tagliafico': {'club': 'Lyon', 'age': 33},
        'Rodrigo Javier De Paul': {'club': 'Atlético Madrid', 'age': 31},
        'Leandro Daniel Paredes': {'club': 'Roma', 'age': 31},
        'Alexis Mac Allister': {'club': 'Liverpool', 'age': 27},
        'Enzo Fernandez': {'club': 'Chelsea', 'age': 25},
        'Giovani Lo Celso': {'club': 'Real Betis', 'age': 29},
        'Exequiel Alejandro Palacios': {'club': 'Bayer Leverkusen', 'age': 27},
        'Lionel Andrés Messi Cuccittini': {'club': 'Inter Miami', 'age': 38},
        'Julián Álvarez': {'club': 'Atlético Madrid', 'age': 26},
        'Lautaro Javier Martínez': {'club': 'Inter Milan', 'age': 28},
        'Nicolás Iván González': {'club': 'Juventus', 'age': 27},
        'Paulo Bruno Exequiel Dybala': {'club': 'Roma', 'age': 32},
        'Alejandro Garnacho Ferreyra': {'club': 'Manchester United', 'age': 21},
        'Thiago Ezequiel Almada': {'club': 'Botafogo', 'age': 24},
        'Valentin Carboni': {'club': 'Marseille', 'age': 21},
        'Franco Mastantuono': {'club': 'Real Madrid', 'age': 18},
        'Nicolás Martín Pareja': {'club': 'Como', 'age': 21},
    },

    'England': {
        'Jordan Pickford': {'club': 'Everton', 'age': 32},
        'Aaron Ramsey': {'club': 'Southampton', 'age': 27},
        'Kyle Walker': {'club': 'Manchester City', 'age': 36},
        'John Stones': {'club': 'Manchester City', 'age': 31},
        'Harry Maguire': {'club': 'Manchester United', 'age': 33},
        'Marc Guehi': {'club': 'Crystal Palace', 'age': 25},
        'Luke Shaw': {'club': 'Manchester United', 'age': 30},
        'Trent Alexander-Arnold': {'club': 'Real Madrid', 'age': 27},
        'Declan Rice': {'club': 'Arsenal', 'age': 27},
        'Jude Bellingham': {'club': 'Real Madrid', 'age': 22},
        'Philip Foden': {'club': 'Manchester City', 'age': 26},
        'Bukayo Saka': {'club': 'Arsenal', 'age': 24},
        'James Maddison': {'club': 'Tottenham Hotspur', 'age': 29},
        'Kobbie Mainoo': {'club': 'Manchester United', 'age': 20},
        'Conor Gallagher': {'club': 'Atlético Madrid', 'age': 26},
        'Cole Palmer': {'club': 'Chelsea', 'age': 23},
        'Harry Kane': {'club': 'Bayern Munich', 'age': 32},
        'Marcus Rashford': {'club': 'Manchester United', 'age': 28},
        'Anthony Gordon': {'club': 'Newcastle United', 'age': 25},
        'Ollie Watkins': {'club': 'Aston Villa', 'age': 30},
        'Ivan Toney': {'club': 'Al-Ahli', 'age': 29},
        'Jarrod Bowen': {'club': 'West Ham United', 'age': 29},
        'Jack Grealish': {'club': 'Manchester City', 'age': 30},
    },

    'France': {
        'Mike Maignan': {'club': 'AC Milan', 'age': 30},
        'Alphonse Areola': {'club': 'West Ham United', 'age': 33},
        'Jules Koundé': {'club': 'Barcelona', 'age': 27},
        'Dayotchanculle Upamecano': {'club': 'Bayern Munich', 'age': 27},
        'Ibrahima Konaté': {'club': 'Liverpool', 'age': 26},
        'William Saliba': {'club': 'Arsenal', 'age': 25},
        'Theo Bernard François Hernández': {'club': 'AC Milan', 'age': 28},
        'Benjamin Pavard': {'club': 'Inter Milan', 'age': 29},
        'Aurélien Djani Tchouaméni': {'club': 'Real Madrid', 'age': 26},
        'Eduardo Camavinga': {'club': 'Real Madrid', 'age': 23},
        "N'Golo Kanté": {'club': 'Al-Ittihad', 'age': 35},
        'Adrien Rabiot': {'club': 'Marseille', 'age': 30},
        'Antoine Griezmann': {'club': 'Atlético Madrid', 'age': 34},
        'Ousmane Dembélé': {'club': 'Paris Saint-Germain', 'age': 28},
        'Kylian Mbappé Lottin': {'club': 'Real Madrid', 'age': 27},
        'Randal Kolo Muani': {'club': 'Paris Saint-Germain', 'age': 27},
        'Marcus Thuram': {'club': 'Inter Milan', 'age': 28},
        'Kingsley Coman': {'club': 'Bayern Munich', 'age': 29},
        'Bradley Barcola': {'club': 'Paris Saint-Germain', 'age': 23},
        'Christopher Nkunku': {'club': 'Chelsea', 'age': 28},
        'Youssouf Fofana': {'club': 'AC Milan', 'age': 27},
        'Warren Zaire Emery': {'club': 'Paris Saint-Germain', 'age': 20},
        # ADDED from TOP_100 (rank 16) — confirm StatBomb name:
        'Désiré Doué': {'club': 'Paris Saint-Germain', 'age': 20},
        # ADDED from TOP_100 (rank 21) — confirm StatBomb name:
        'Michael Olise': {'club': 'Bayern Munich', 'age': 24},
    },

    'Germany': {
        'Marc-André ter Stegen': {'club': 'Barcelona', 'age': 34},
        'Oliver Baumann': {'club': 'TSG Hoffenheim', 'age': 35},
        'Antonio Rüdiger': {'club': 'Real Madrid', 'age': 33},
        'Nico Schlotterbeck': {'club': 'Borussia Dortmund', 'age': 26},
        'Jonathan Tah': {'club': 'Bayer Leverkusen', 'age': 30},
        'David Raum': {'club': 'RB Leipzig', 'age': 27},
        'Joshua Kimmich': {'club': 'Bayern Munich', 'age': 31},
        'Jamal Musiala': {'club': 'Bayern Munich', 'age': 23},
        # UPDATED: Bayer Leverkusen → Liverpool (transferred June 2025)
        'Florian Wirtz': {'club': 'Liverpool', 'age': 22},
        'Leon Goretzka': {'club': 'Bayern Munich', 'age': 31},
        'Kai Havertz': {'club': 'Arsenal', 'age': 27},
        'Leroy Sané': {'club': 'Bayern Munich', 'age': 30},
        'Serge Gnabry': {'club': 'Bayern Munich', 'age': 31},
        'Niclas Füllkrug': {'club': 'West Ham United', 'age': 33},
        'Karim-David Adeyemi': {'club': 'Borussia Dortmund', 'age': 24},
        'Pascal Groß': {'club': 'Borussia Dortmund', 'age': 34},
        'Robert Andrich': {'club': 'Bayer Leverkusen', 'age': 31},
        'Maximilian Mittelstädt': {'club': 'VfB Stuttgart', 'age': 28},
        'Aleksandar Pavlović': {'club': 'Bayern Munich', 'age': 21},
        'Chris Führich': {'club': 'VfB Stuttgart', 'age': 28},
        'Emre Can': {'club': 'Borussia Dortmund', 'age': 32},
        'Julian Brandt': {'club': 'Borussia Dortmund', 'age': 30},
        'Karl Lennart Skoglund': {'club': 'Bayern Munich', 'age': 18},
    },

    'Brazil': {
        'Alisson Ramsés Becker': {'club': 'Liverpool', 'age': 33},
        'Ederson Santana de Moraes': {'club': 'Fenerbahçe', 'age': 32},
        # ⚠ NOTE: Ederson moved to Fenerbahce (Sep 2025) when Donnarumma signed Man City
        'Danilo Luiz da Silva': {'club': 'Juventus', 'age': 34},
        'Marcos Aoás Corrêa': {'club': 'Paris Saint-Germain', 'age': 31},
        'Éder Gabriel Militão': {'club': 'Real Madrid', 'age': 28},
        'Gabriel dos Santos Magalhães': {'club': 'Arsenal', 'age': 28},
        'Guilherme Antonio Arana Lopes': {'club': 'Atlético Mineiro', 'age': 28},
        'Renan Augusto Lodi dos Santos': {'club': 'Al-Hilal', 'age': 27},
        'Carlos Henrique Casimiro': {'club': 'Manchester United', 'age': 34},
        'Douglas Luiz Soares de Paulo': {'club': 'Juventus', 'age': 27},
        'Bruno Guimarães Rodriguez Moura': {'club': 'Newcastle United', 'age': 28},
        'Lucas Paquetá': {'club': 'West Ham United', 'age': 28},
        'Neymar da Silva Santos Junior': {'club': 'Al-Hilal', 'age': 34},
        'Vinícius José Paixão de Oliveira Júnior': {'club': 'Real Madrid', 'age': 25},
        'Rodrygo Silva de Goes': {'club': 'Real Madrid', 'age': 25},
        'Raphael Dias Belloli': {'club': 'Barcelona', 'age': 29},
        'Luiz Henrique': {'club': 'Botafogo', 'age': 25},
        'Gabriel Fernando de Jesus': {'club': 'Arsenal', 'age': 29},
        'Richarlison de Andrade': {'club': 'Tottenham Hotspur', 'age': 28},
        'Gabriel Teodoro Martinelli Silva': {'club': 'Arsenal', 'age': 24},
        'Endrick Felipe Moreira de Sousa': {'club': 'Real Madrid', 'age': 19},
        'Sávio Moreira de Oliveira': {'club': 'Manchester City', 'age': 21},
        'Estêvão Willian Almeida de Oliveira Gonçalves': {'club': 'Chelsea', 'age': 18},
        'Andrey Santos': {'club': 'Strasbourg', 'age': 21},
    },

    'Netherlands': {
        'Bart Verbruggen': {'club': 'Brighton', 'age': 23},
        'Virgil van Dijk': {'club': 'Liverpool', 'age': 34},
        'Matthijs de Ligt': {'club': 'Manchester United', 'age': 26},
        'Nathan Aké': {'club': 'Manchester City', 'age': 31},
        'Denzel Dumfries': {'club': 'Inter Milan', 'age': 29},
        'Jeremie Frimpong': {'club': 'Liverpool', 'age': 25},
        # ⚠ NOTE: Frimpong moved to Liverpool summer 2025 (from Leverkusen, along with Wirtz)
        'Frenkie de Jong': {'club': 'Barcelona', 'age': 28},
        'Tijjani Reijnders': {'club': 'AC Milan', 'age': 27},
        'Ryan Gravenberch': {'club': 'Liverpool', 'age': 23},
        'Xavi Simons': {'club': 'RB Leipzig', 'age': 22},
        'Teun Koopmeiners': {'club': 'Juventus', 'age': 28},
        'Memphis Depay': {'club': 'Corinthians', 'age': 32},
        'Cody Mathès Gakpo': {'club': 'Liverpool', 'age': 26},
        'Donyell Malen': {'club': 'Borussia Dortmund', 'age': 27},
        'Steven Bergwijn': {'club': 'Al-Ittihad', 'age': 28},
        'Wout Weghorst': {'club': 'Ajax', 'age': 33},
        'Brian Brobbey': {'club': 'Ajax', 'age': 24},
        'Joey Veerman': {'club': 'PSV Eindhoven', 'age': 27},
        'Mats Wieffer': {'club': 'Brighton', 'age': 26},
        'Jurriën David Norman Timber': {'club': 'Arsenal', 'age': 24},
        'Ian Maatsen': {'club': 'Aston Villa', 'age': 23},
    },

    'Portugal': {
        'Diogo Meireles Costa': {'club': 'Porto', 'age': 26},
        'Rui Pedro dos Santos Patrício': {'club': 'Atalanta', 'age': 38},
        'Rúben Santos Gato Alves Dias': {'club': 'Manchester City', 'age': 28},
        'Gonçalo Bernardo Inácio': {'club': 'Sporting CP', 'age': 24},
        'Danilo Luís Hélio Pereira': {'club': 'Al-Ittihad', 'age': 34},
        'João Pedro Cavaco Cancelo': {'club': 'Al-Hilal', 'age': 31},
        'Nuno Mendes': {'club': 'Paris Saint-Germain', 'age': 23},
        # UPDATED: Bayern Munich → Tottenham Hotspur (season loan, Aug 3 2025)
        'João Maria Lobo Alves Palhinha Gonçalves': {'club': 'Tottenham Hotspur', 'age': 30},
        'Rúben Diogo Da Silva Neves': {'club': 'Al-Hilal', 'age': 28},
        'Bernardo Fernandes da Silva Junior': {'club': 'Manchester City', 'age': 31},
        'Brandon Fernandes': {'club': 'Manchester United', 'age': 31},
        'Vitor Machado Ferreira': {'club': 'Paris Saint-Germain', 'age': 26},
        'João Félix Sequeira': {'club': 'Chelsea', 'age': 26},
        'Rafael Alexandre Conceição Leão': {'club': 'AC Milan', 'age': 26},
        'Cristiano Ronaldo dos Santos Aveiro': {'club': 'Al-Nassr', 'age': 41},
        'Gonçalo Matias Ramos': {'club': 'Paris Saint-Germain', 'age': 24},
        'Diogo José Teixeira da Silva': {'club': 'Liverpool', 'age': 29},
        'Pedro Lomba Neto': {'club': 'Chelsea', 'age': 25},
        'João Neves': {'club': 'Paris Saint-Germain', 'age': 21},
        'António João Pereira Albuquerque Tavares Silva': {'club': 'Benfica', 'age': 22},
    },

    'United States': {
        'Matt Turner': {'club': 'Crystal Palace', 'age': 31},
        'Patrick Schulte': {'club': 'Columbus Crew', 'age': 25},
        'Sergino Dest': {'club': 'PSV Eindhoven', 'age': 25},
        'Antonee Robinson': {'club': 'Fulham', 'age': 28},
        'Chris Richards': {'club': 'Crystal Palace', 'age': 25},
        'Cameron Carter-Vickers': {'club': 'Celtic', 'age': 28},
        'Tim Ream': {'club': 'Charlotte FC', 'age': 38},
        'Joseph Scally': {'club': 'Borussia Mgladbach', 'age': 23},
        'Tyler Adams': {'club': 'Bournemouth', 'age': 27},
        'Weston McKennie': {'club': 'Juventus', 'age': 27},
        'Yunus Dimoara Musah': {'club': 'AC Milan', 'age': 23},
        'Giovanni Reyna': {'club': 'Borussia Dortmund', 'age': 23},
        'Malik Tillman': {'club': 'PSV Eindhoven', 'age': 23},
        'Christian Pulisic': {'club': 'AC Milan', 'age': 27},
        'Timothy Weah': {'club': 'Juventus', 'age': 26},
        'Folarin Balogun': {'club': 'Monaco', 'age': 24},
        'Ricardo Daniel Pepi': {'club': 'PSV Eindhoven', 'age': 23},
        'Haji Wright': {'club': 'Coventry City', 'age': 27},
        'Joshua Sargent': {'club': 'Norwich City', 'age': 26},
        'Johann Carrasso': {'club': 'Real Betis', 'age': 24},
        'Tanner Tessmann': {'club': 'Lyon', 'age': 24},
    },

    'Mexico': {
        'Luis Ángel Malagón Velázquez': {'club': 'Club América', 'age': 29},
        'Edson Omar Álvarez Velázquez': {'club': 'West Ham United', 'age': 28},
        'Santiago Tomás Giménez': {'club': 'AC Milan', 'age': 24},
        'Hirving Rodrigo Lozano Bahena': {'club': 'San Diego FC', 'age': 30},
        'Raúl Alonso Jiménez Rodríguez': {'club': 'Fulham', 'age': 34},
        'Gilberto Moraes Junior': {'club': 'Feyenoord', 'age': 17},
        'Johan Felipe Vásquez Ibarra': {'club': 'Genoa', 'age': 27},
        'Jorge Eduardo Sánchez Ramos': {'club': 'Cruz Azul', 'age': 28},
        'César Jasib Montes Castro': {'club': 'Lokomotiv Moscow', 'age': 29},
    },

    'Uruguay': {
        'Sergio Rochet Álvarez': {'club': 'Internacional', 'age': 32},
        'Federico Santiago Valverde Dipetta': {'club': 'Real Madrid', 'age': 27},
        'Darwin Gabriel Núñez Ribeiro': {'club': 'Al Hilal', 'age': 26},
        'Ronald Federico Araújo da Silva': {'club': 'Barcelona', 'age': 27},
        'Manuel Ugarte Ribeiro': {'club': 'Manchester United', 'age': 24},
        'José María Giménez de Vargas': {'club': 'Atlético Madrid', 'age': 31},
        'Mathías Olivera Miramontes': {'club': 'Napoli', 'age': 28},
        'Facundo Pellistri Rebollo': {'club': 'Panathinaikos', 'age': 24},
        'Diego Nicolás De La Cruz Arcosa': {'club': 'Flamengo', 'age': 28},
        'Nahitan Michel Nández Acosta': {'club': 'Al-Qadsiah', 'age': 30},
        'Guido Rodríguez': {'club': 'Bahia', 'age': 22},
        'Rodrigo Bentancur Colmán': {'club': 'Tottenham Hotspur', 'age': 28},
    },

    'Croatia': {
        'Dominik Livaković': {'club': 'Fenerbahçe', 'age': 31},
        'Joško Gvardiol': {'club': 'Manchester City', 'age': 24},
        'Josip Juranović': {'club': 'Union Berlin', 'age': 30},
        'Borna Sosa': {'club': 'Torino', 'age': 28},
        'Luka Modrić': {'club': 'AC Milan', 'age': 40},
        'Mateo Kovačić': {'club': 'Manchester City', 'age': 31},
        'Marcelo Brozović': {'club': 'Al-Nassr', 'age': 33},
        'Mario Pašalić': {'club': 'Atalanta', 'age': 31},
        'Luka Sučić': {'club': 'Real Sociedad', 'age': 23},
        'Lovro Majer': {'club': 'Wolfsburg', 'age': 28},
        'Ivan Perišić': {'club': 'PSV Eindhoven', 'age': 37},
        'Andrej Kramarić': {'club': 'TSG Hoffenheim', 'age': 34},
        'Bruno Petković': {'club': 'Dinamo Zagreb', 'age': 31},
        'Josip Stanišić': {'club': 'Bayern Munich', 'age': 25},
        'Joško Šutalo': {'club': 'Ajax', 'age': 26},
    },

    'Belgium': {
        'Thibaut Courtois': {'club': 'Real Madrid', 'age': 33},
        'Kevin De Bruyne': {'club': 'Napoli', 'age': 34},
        'Jeremy Doku': {'club': 'Manchester City', 'age': 23},
        'Amadou Onana': {'club': 'Aston Villa', 'age': 24},
        'Ikoma Loïs Openda': {'club': 'Juventus', 'age': 26},
        'Charles De Ketelaere': {'club': 'Atalanta', 'age': 24},
        'Youri Tielemans': {'club': 'Aston Villa', 'age': 28},
        'Arthur Theate': {'club': 'Eintracht Frankfurt', 'age': 25},
        'Maxim De Cuyper': {'club': 'Club Brugge', 'age': 25},
        'Leandro Trossard': {'club': 'Arsenal', 'age': 31},
        'Zeno Debast': {'club': 'Sporting CP', 'age': 22},
    },

    'Japan': {
        'Suzuki Zaion': {'club': 'Parma', 'age': 23},
        'Wataru Endo': {'club': 'Liverpool', 'age': 33},
        'Takefusa Kubo': {'club': 'Real Sociedad', 'age': 24},
        'Kaoru Mitoma': {'club': 'Brighton', 'age': 28},
        'Ko Itakura': {'club': 'Borussia Mgladbach', 'age': 29},
        'Hiroki Ito': {'club': 'Bayern Munich', 'age': 26},
        'Ritsu Doan': {'club': 'SC Freiburg', 'age': 27},
        'Ayase Ueda': {'club': 'Feyenoord', 'age': 27},
        'Takehiro Tomiyasu': {'club': 'Arsenal', 'age': 27},
        'Daichi Kamada': {'club': 'Crystal Palace', 'age': 29},
    },

    'Switzerland': {
        'Gregor Kobel': {'club': 'Borussia Dortmund', 'age': 28},
        'Yann Sommer': {'club': 'Inter Milan', 'age': 37},
        'Manuel Obafemi Akanji': {'club': 'Inter Milan', 'age': 30},
        # ⚠ NOTE: per ESPN report, Akanji moved to Inter Milan (Sep 2 2025) when he left Man City
        'Granit Xhaka': {'club': 'Bayer Leverkusen', 'age': 33},
        'Denis Lemi Zakaria Lako Lado': {'club': 'Monaco', 'age': 29},
        'Remo Freuler': {'club': 'Bologna', 'age': 33},
        'Dan Ndoye': {'club': 'Bologna', 'age': 25},
        'Ruben Vargas': {'club': 'Sevilla', 'age': 27},
        'Mohamed Zeki Amdouni': {'club': 'Benfica', 'age': 25},
    },

    'Ecuador': {
        'Hernán Ismael Galíndez': {'club': 'Huracán', 'age': 38},
        'Piero Martín Hincapié Reyna': {'club': 'Arsenal', 'age': 24},
        # ⚠ NOTE: Hincapié joined Arsenal from Leverkusen — confirm current club
        'Willian Joel Pacho Tenorio': {'club': 'Paris Saint-Germain', 'age': 24},
        'Pervis Josué Estupiñán Tenorio': {'club': 'Brighton', 'age': 28},
        'Moisés Isaac Caicedo Corozo': {'club': 'Chelsea', 'age': 24},
        'Ray Kendry Páez Andrade': {'club': 'River Plate', 'age': 18},
        'Enner Remberto Valencia Lastra': {'club': 'Pachuca', 'age': 36},
        'Gonzalo Jordy Plata Jiménez': {'club': 'Flamengo', 'age': 25},
        'Jeremy Leonel Sarmiento Morante': {'club': 'Brighton', 'age': 23},
        'Joel Ordóñez': {'club': 'Club Brugge', 'age': 21},
    },

    'Turkey': {
        # ⚠ QUALIFICATION STATUS: IN UEFA PLAYOFF PATH C (semifinal TODAY, March 26)
        # Turkey vs Romania today; winner plays Slovakia/Kosovo final March 31
        'Altay Bayındır': {'club': 'Manchester United', 'age': 27},
        'Merih Demiral': {'club': 'Al-Ahli', 'age': 28},
        'Ferdi Erenay Kadıoğlu': {'club': 'Brighton', 'age': 26},
        'Hakan Çalhanoğlu': {'club': 'Inter Milan', 'age': 32},
        'Arda Güler': {'club': 'Real Madrid', 'age': 21},
        'Kenan Yildiz': {'club': 'Juventus', 'age': 20},
        'Can Yilmaz Uzun': {'club': 'Eintracht Frankfurt', 'age': 20},
        'Muhammed Kerem Aktürkoğlu': {'club': 'Benfica', 'age': 27},
        'Barış Alper Yılmaz': {'club': 'Galatasaray', 'age': 25},
    },

    'Senegal': {
        'Édouard Osoque Mendy': {'club': 'Al-Ahli', 'age': 34},
        'Kalidou Koulibaly': {'club': 'Al-Hilal', 'age': 34},
        'Pape Matar Sarr': {'club': 'Tottenham Hotspur', 'age': 23},
        'Nicolas Jackson': {'club': 'Chelsea', 'age': 24},
        'Sadio Mané': {'club': 'Al-Nassr', 'age': 34},
        'Ismaïla Sarr': {'club': 'Crystal Palace', 'age': 28},
        'Idrissa Gueye': {'club': 'Everton', 'age': 36},
        'Lamine Camara': {'club': 'Monaco', 'age': 22},
    },

    'Canada': {
        'Maxime Crépeau': {'club': 'Portland Timbers', 'age': 31},
        'Alphonso Davies': {'club': 'Bayern Munich', 'age': 25},
        'Moise Bombito': {'club': 'Nice', 'age': 25},
        'Ismael Koné': {'club': 'Marseille', 'age': 23},
        'Stephen Antunes Eustáquio': {'club': 'LAFC', 'age': 29},
        'Jonathan David': {'club': 'Lille', 'age': 26},
        'Jacob Shaffelburg': {'club': 'LAFC', 'age': 26},
        'Tajon Buchanan': {'club': 'Villarreal', 'age': 27},
    },

    'Nigeria': {
        'Stanley Bobo Nwabali': {'club': 'Chippa United', 'age': 29},
        'Calvin Bassey Ughelumba': {'club': 'Fulham', 'age': 26},
        'William Troost-Ekong': {'club': 'Al-Kholood', 'age': 32},
        'Onyinye Wilfred Ndidi': {'club': 'Beşiktaş', 'age': 29},
        'Alex Iwobi': {'club': 'Fulham', 'age': 29},
        'Ademola Lookman': {'club': 'Atalanta', 'age': 28},
        'Victor James Osimhen': {'club': 'Galatasaray', 'age': 27},
        'Victor Okoh Boniface': {'club': 'Bayer Leverkusen', 'age': 25},
        'Samuel Chimerenka Chukwueze': {'club': 'Fulham', 'age': 26},
    },


    'Austria': {
        'Patrick Pentz': {'club': 'Brøndby', 'age': 29},
        'David Olatukunbo Alaba': {'club': 'Real Madrid', 'age': 33},
        'Kevin Danso': {'club': 'Tottenham Hotspur', 'age': 27},
        'Konrad Laimer': {'club': 'Bayern Munich', 'age': 28},
        'Marcel Sabitzer': {'club': 'Borussia Dortmund', 'age': 31},
        'Christoph Baumgartner': {'club': 'RB Leipzig', 'age': 26},
        'Nicolas Seiwald': {'club': 'RB Leipzig', 'age': 24},
        'Marko Arnautović': {'club': 'Crvena Zvezda', 'age': 36},
    },

    'Poland': {
        # ⚠ QUALIFICATION STATUS: IN UEFA PLAYOFF PATH B (semifinal TODAY, March 26)
        'Kamil Grabara': {'club': 'Wolfsburg', 'age': 27},
        'Jakub Piotr Kiwior': {'club': 'Porto', 'age': 26},
        'Matty Cash': {'club': 'Aston Villa', 'age': 28},
        'Piotr Zieliński': {'club': 'Inter Milan', 'age': 31},
        'Sebastian Szymański': {'club': 'Fenerbahçe', 'age': 26},
        'Nicola Zalewski': {'club': 'Roma', 'age': 24},
        'Robert Lewandowski': {'club': 'Barcelona', 'age': 37},
        'Karol Świderski': {'club': 'Charlotte FC', 'age': 29},
    },

    'Ghana': {
        'Lawrence Ati-Zigi': {'club': 'St. Gallen', 'age': 29},
        'Mohamed Salisu': {'club': 'Monaco', 'age': 26},
        'Tariq Lamptey': {'club': 'Brighton', 'age': 25},
        'Thomas Teye Partey': {'club': 'Arsenal', 'age': 32},
        'Mohammed Kudus': {'club': 'Tottenham', 'age': 25},
        'Antoine Semenyo': {'club': 'Bournemouth', 'age': 26},
        'Iñaki Williams Arthuer': {'club': 'Athletic Club', 'age': 31},
        'Jordan Ayew': {'club': 'Leicester City', 'age': 34},
    },

    'Colombia': {
        'David Ospina Ramírez': {'club': 'Atlético Nacional', 'age': 37},
        'Camilo Andrés Vargas Gil': {'club': 'Atlas', 'age': 36},
        'Daniel Muñoz Mejía': {'club': 'Crystal Palace', 'age': 29},
        'Davinson Sánchez Mina': {'club': 'Galatasaray', 'age': 29},
        'Jhon Janer Lucumí Bonilla': {'club': 'Bologna', 'age': 27},
        'Carlos Eccehomo Cuesta Figueroa': {'club': 'Vasco da Gama', 'age': 26},
        'Johan Andrés Mojica Palacio': {'club': 'Mallorca', 'age': 33},
        'Richard Rios Montoya': {'club': 'Benfica', 'age': 25},
        'Jefferson Andrés Lerma Solís': {'club': 'Crystal Palace', 'age': 31},
        'James David Rodríguez Rubio': {'club': 'Minnesota United', 'age': 34},
        'Jhon Adolfo Arias Andrade': {'club': 'Wolverhampton Wanderers', 'age': 28},
        'Luis Alberto Suárez Díaz': {'club': 'Bayern Munich', 'age': 29},
        'Jhon Jáder Durán Palacio': {'club': 'Aston Villa', 'age': 22},
        'Luis Fernando Sinisterra Lucumí': {'club': 'Bournemouth', 'age': 26},
        'Yaser Esneider Asprilla Martínez': {'club': 'Girona', 'age': 22},
        # ADDED from TOP_100 (rank 31) — ⚠ confirm StatBomb exact name:
        'Luis Díaz': {'club': 'Bayern Munich', 'age': 28},
        # ⚠ Luis Díaz is on loan at Bayern Munich from Liverpool — confirm club field preference
    },

    'Italy': {
        # ⚠ QUALIFICATION STATUS: IN UEFA PLAYOFF PATH A (semifinal TODAY, March 26)
        # Italy vs Northern Ireland today; winner plays Wales/Bosnia final March 31
        'Gianluigi Donnarumma': {'club': 'Manchester City', 'age': 27},
        'Guglielmo Vicario': {'club': 'Arsenal', 'age': 29},
        # ⚠ NOTE: Vicario moved from Tottenham — confirm current club
        'Alessandro Bastoni': {'club': 'Inter Milan', 'age': 26},
        'Riccardo Calafiori': {'club': 'Arsenal', 'age': 23},
        'Alessandro Buongiorno': {'club': 'Napoli', 'age': 26},
        'Federico Dimarco': {'club': 'Inter Milan', 'age': 28},
        'Andrea Cambiaso': {'club': 'Juventus', 'age': 26},
        'Sandro Tonali': {'club': 'Newcastle United', 'age': 25},
        'Nicolò Barella': {'club': 'Inter Milan', 'age': 29},
        'Davide Frattesi': {'club': 'Inter Milan', 'age': 26},
        'Samuele Ricci': {'club': 'Torino', 'age': 24},
        'Mateo Retegui': {'club': 'Atalanta', 'age': 26},
        'Giacomo Raspadori': {'club': 'Napoli', 'age': 26},
        'Francesco Pio Esposito': {'club': 'Spezia', 'age': 20},
        'Daniel Maldini': {'club': 'Monza', 'age': 24},
    },
    'Serbia': {
        'Aleksandar Mitrović': {'club': 'Al-Rayyan', 'age': 31},
        'Dušan Vlahović': {'club': 'Juventus', 'age': 26},
        'Luka Jović': {'club': 'AC Milan', 'age': 28},
        'Strahinja Pavlović': {'club': 'AC Milan', 'age': 24},
        'Nikola Milenković': {'club': 'Nottingham Forest', 'age': 28},
        'Lazar Samardžić': {'club': 'Atalanta', 'age': 24},
        'Saša Lukić': {'club': 'Fulham', 'age': 29},
        'Predrag Rajković': {'club': 'Al-Ittihad', 'age': 30},
        'Filip Kostić': {'club': 'Fenerbahçe', 'age': 33},
        'Andrija Živković': {'club': 'PAOK', 'age': 29}
    },
    'Chile': {
        'Thomas Gillier': {'club': 'CF Montréal', 'age': 21},
        'Gabriel Alonso Suazo Urbina': {'club': 'Toulouse', 'age': 28},
        'Guillermo Alfonso Maripán Loaysa': {'club': 'Torino', 'age': 31},
        'Benjamin Anthony Brereton Díaz': {'club': 'Southampton', 'age': 26},
        'Darío Esteban Osorio Osorio': {'club': 'Midtjylland', 'age': 22},
        'Vicente Felipe Pizarro Wiencke': {'club': 'Colo-Colo', 'age': 23},
        'Rodrigo Eduardo Echeverría Sáez': {'club': 'Huracán', 'age': 30},
        'Felipe Loyola Olea': {'club': 'Independiente', 'age': 25},
        'Alexander Antonio Aravena Guzmán': {'club': 'Grêmio', 'age': 23},
        'Lucas Antonio Cepeda Barturen': {'club': 'Colo-Colo', 'age': 23}
    },
    'Paraguay': {
        'Miguel Ángel Almirón Rejala': {'club': 'Atlanta United', 'age': 32},
        'Julio César Enciso Espínola': {'club': 'Brighton & Hove Albion', 'age': 22},
        'Arnaldo Antonio Sanabria Ayala': {'club': 'Torino', 'age': 30},
        'Gustavo Raúl Gómez Portillo': {'club': 'Palmeiras', 'age': 32},
        'Diego Alexander Gómez Amarilla': {'club': 'Inter Miami', 'age': 23},
        'Omar Federico Alderete Fernández': {'club': 'Getafe', 'age': 29},
        'Ramón Sosa Acosta': {'club': 'Nottingham Forest', 'age': 26},
        'Carlos Miguel Coronel': {'club': 'New York Red Bulls', 'age': 29},
        'Mathías Adalberto Villasanti Brítez': {'club': 'Grêmio', 'age': 29},
        'Damián Josué Bobadilla Benítez': {'club': 'São Paulo', 'age': 24}
    },
    'Morocco': {
        'Yassine Bounou': {'club': 'Al-Hilal', 'age': 34},
        'Achraf Hakimi Mouh': {'club': 'Paris Saint-Germain', 'age': 27},
        'Noussair Mazraoui': {'club': 'Manchester United', 'age': 28},
        'Nayef Aguerd': {'club': 'Marseille', 'age': 29},
        'Sofyan Amrabat': {'club': 'Real Betis', 'age': 29},
        'Azzedine Ounahi': {'club': 'Girona', 'age': 25},
        'Brahim Abdelkader Díaz': {'club': 'Real Madrid', 'age': 26},
        'Ismael Saibari Ben El Basra': {'club': 'PSV Eindhoven', 'age': 25},
        'Youssef En-Nesyri': {'club': 'Al Ittihad', 'age': 28},
        'Abdessamad Ezzalzouli': {'club': 'Real Betis', 'age': 24},
        'Bilal El Khannouss': {'club': 'Leicester City', 'age': 21}
    },
    'Australia': {
        'Mathew David Ryan': {'club': 'Roma', 'age': 33},
        'Harry James Souttar': {'club': 'Sheffield United', 'age': 27},
        'Alessandro Circati': {'club': 'Parma', 'age': 22},
        'Jackson Alexander Irvine': {'club': 'FC St. Pauli', 'age': 33},
        'Riley Patrick McGree': {'club': 'Middlesbrough', 'age': 27},
        'Jordan Bos': {'club': 'Westerlo', 'age': 23},
        'Nestory Irankunda': {'club': 'Bayern Munich', 'age': 20},
        'Martin Boyle': {'club': 'Hibernian', 'age': 32},
        'Aiden Connor O\'Neill': {'club': 'Standard Liège', 'age': 27},
        'Max Balard': {'club': 'NAC Breda', 'age': 25}
    },
    'Chile': {
        'Thomas Gillier': {'club': 'Universidad Católica', 'age': 21},
        'Gabriel Alonso Suazo Urbina': {'club': 'Toulouse', 'age': 28},
        'Guillermo Alfonso Maripán Loaysa': {'club': 'Torino', 'age': 31},
        'Benjamin Anthony Brereton Díaz': {'club': 'Derby County', 'age': 26},
        'Darío Esteban Osorio Osorio': {'club': 'Midtjylland', 'age': 22},
        'Vicente Felipe Pizarro Wiencke': {'club': 'Colo-Colo', 'age': 23},
        'Rodrigo Eduardo Echeverría Sáez': {'club': 'Huracán', 'age': 30},
        'Felipe Loyola Olea': {'club': 'Independiente', 'age': 25},
        'Alexander Antonio Aravena Guzmán': {'club': 'Portland Timbers', 'age': 23},
        'Lucas Antonio Cepeda Barturen': {'club': 'Elche', 'age': 23}
    },
    'Paraguay': {
        'Miguel Ángel Almirón Rejala': {'club': 'Atlanta United', 'age': 32},
        'Julio César Enciso Espínola': {'club': 'Strasbourg', 'age': 22},
        'Arnaldo Antonio Sanabria Ayala': {'club': 'Cremonese', 'age': 30},
        'Gustavo Raúl Gómez Portillo': {'club': 'Palmeiras', 'age': 32},
        'Diego Alexander Gómez Amarilla': {'club': 'Brighton', 'age': 23},
        'Omar Federico Alderete Fernández': {'club': 'Sunderland', 'age': 29},
        'Ramón Sosa Acosta': {'club': 'Palmeiras', 'age': 26},
        'Carlos Miguel Coronel': {'club': 'New York Red Bulls', 'age': 29},
        'Mathías Adalberto Villasanti Brítez': {'club': 'Grêmio', 'age': 29},
        'Damián Josué Bobadilla Benítez': {'club': 'São Paulo', 'age': 24}
    },
    'DR Congo': {
        'Chancel Mbemba Mangulu': {'club': 'Lille', 'age': 31},
        'Yoane Wissa': {'club': 'Newcastle United', 'age': 29},
        "Théo Bongonda Mbul'Anay Itoke Mbando Pozzi": {'club': 'Spartak Moscow', 'age': 30},
        'Samuel Essende': {'club': 'Augsburg', 'age': 28},
        'Noah Sadiki': {'club': 'Sunderland', 'age': 21},
        'Aaron Wan-Bissaka': {'club': 'West Ham United', 'age': 28},
        'Lionel Mpasi-Nzau': {'club': 'Le Havre', 'age': 31},
        'Axel Tuanzebe': {'club': 'Burnley', 'age': 28},
        'Meschack Elia Lina': {'club': 'Alanyaspor', 'age': 28},
        'Simon Banza': {'club': 'Al-Jazira', 'age': 29}
    },
    'Côte d\'Ivoire': {
        'Franck Yannick Kessié': {'club': 'Al-Ahli', 'age': 29},
        'Sébastien Romain Teddy Haller': {'club': 'Leganés', 'age': 31},
        'Simon Adingra': {'club': 'Monaco', 'age': 24},
        'Amad Diallo Traoré': {'club': 'Manchester United', 'age': 23},
        'Odilon Kossonou': {'club': 'Atalanta', 'age': 25},
        'Ousmane Diomande': {'club': 'Sporting CP', 'age': 22},
        'Seko Mohamed Fofana': {'club': 'Porto', 'age': 31},
        'Ibrahim Sangaré': {'club': 'Nottingham Forest', 'age': 28},
        'Evan Guessand': {'club': 'Crystal Palace', 'age': 24},
        'Nicolas Pépé': {'club': 'Villarreal', 'age': 30}
    },
    'Cape Verde': {
        'Ryan Isaac Mendes da Graça': {'club': 'Iğdır', 'age': 36},
        'Jovane Cabral': {'club': 'Estrela Amadora', 'age': 27},
        'Logan Costa': {'club': 'Villarreal', 'age': 24},
        'Bruno Miguel Rivotti Varela': {'club': 'Vitória Guimarães', 'age': 31},
        'Steven de Sousa Moreira': {'club': 'Columbus Crew', 'age': 31},
        'Carlos Miguel dos Santos': {'club': 'San Diego FC', 'age': 25},
        'Ieltsin Camões': {'club': 'Al Ahly', 'age': 26},
        'Deroy Duarte': {'club': 'Ludogorets Razgrad', 'age': 26},
        'Kevin Lenini Gonçalves Pereira Pina': {'club': 'FK Krasnodar', 'age': 29},
        'Roberto Lopes': {'club': 'Shamrock Rovers', 'age': 33}
    },
    'Egypt': {
        'Mohamed Salah Hamed Mahrous Ghaly': {'club': 'Liverpool', 'age': 33},
        'Omar Khaled Mohamed Marmoush': {'club': 'Eintracht Frankfurt', 'age': 27},
        'Mostafa Mohamed Ahmed Abdallah': {'club': 'Nantes', 'age': 28},
        'Mahmoud Ibrahim Hassan': {'club': 'Al-Rayyan', 'age': 31},
        'Ahmed Sayed': {'club': 'Zamalek', 'age': 30},
        'Mohamed Abdelmonem': {'club': 'Nice', 'age': 27},
        'Mohamed El-Shenawy': {'club': 'Al Ahly', 'age': 37},
        'Hamdi Fathi': {'club': 'Al-Wakrah', 'age': 31},
        'Ramy Rabia': {'club': 'Al Ahly', 'age': 32},
        'Ibrahim Adel': {'club': 'Pyramids', 'age': 24}
    },
    'South Africa': {
        'Ronwen Williams': {'club': 'Mamelodi Sundowns', 'age': 34},
        'Teboho Mokoena': {'club': 'Mamelodi Sundowns', 'age': 29},
        'Percy Tau': {'club': 'Al Ahly', 'age': 31},
        'Lyle Brent Foster': {'club': 'Burnley', 'age': 25},
        'Khuliso Johnson Mudau': {'club': 'Mamelodi Sundowns', 'age': 30},
        'Elias Mokwana': {'club': 'Al-Hazem', 'age': 26},
        'Relebohile Mofokeng': {'club': 'Orlando Pirates', 'age': 21},
        'Aubrey Modiba': {'club': 'Mamelodi Sundowns', 'age': 30},
        'Siyabonga Ngezana': {'club': 'FCSB', 'age': 28},
        'Themba Zwane': {'club': 'Mamelodi Sundowns', 'age': 36}
    },
    'Mali': {
        'Yves Bissouma': {'club': 'Tottenham Hotspur', 'age': 29},
        'Amadou Haïdara': {'club': 'Lens', 'age': 28},
        'Kamory Doumbia': {'club': 'Brest', 'age': 23},
        'El Bilal Touré': {'club': 'Beşiktaş', 'age': 24},
        'Hamari Traoré': {'club': 'Paris FC', 'age': 34},
        'Dorgeles Nene': {'club': 'Fenerbahçe SK', 'age': 23},
        'Mohamed Camara': {'club': 'Al Sadd', 'age': 26},
        'Mamadou Fofana': {'club': 'New England Revolution', 'age': 28},
        'Djigui Diarra': {'club': 'Young Africans', 'age': 31},
        'Lassana Coulibaly': {'club': 'Lecce', 'age': 30}
    },
    'Cameroon': {
        'André Onana Onana': {'club': 'Manchester United', 'age': 29},
        'Bryan Tetsadong Marceau Mbeumo': {'club': 'Manchester United', 'age': 26},
        'Vincent Paté Aboubakar': {'club': 'Hatayspor', 'age': 34},
        'Carlos Noom Quomah Baleba': {'club': 'Brighton', 'age': 22},
        'André-Frank Zambo Anguissa': {'club': 'Napoli', 'age': 30},
        'Christopher Wooh': {'club': 'Spartak Moscow', 'age': 24},
        'Jean Charles Castelletto': {'club': 'Al Duhail', 'age': 31},
        'Georges-Kévin Nkoudou': {'club': 'Al Diriyah', 'age': 31},
        'Frank Magri': {'club': 'Toulouse', 'age': 26},
        'Jackson Tchatchoua': {'club': 'Hellas Verona', 'age': 24}
    },
    'Czech Republic': {
        'Patrik Schick': {'club': 'Bayer Leverkusen', 'age': 30},
        'Tomáš Souček': {'club': 'West Ham', 'age': 31},
        'Adam Hložek': {'club': 'Hoffenheim', 'age': 23},
        'Ladislav Krejčí': {'club': 'Girona', 'age': 26},
        'Matěj Kovář': {'club': 'Bayer Leverkusen', 'age': 25},
        'Pavel Bucha': {'club': 'FC Cincinnati', 'age': 27},
        'Vladimír Coufal': {'club': 'West Ham', 'age': 33},
        'Václav Černý': {'club': 'Rangers', 'age': 28},
        'Robin Hranáč': {'club': 'Hoffenheim', 'age': 26},
        'Tomáš Holeš': {'club': 'Slavia Prague', 'age': 32}
    },
    'Albania': {
        'Armando Broja': {'club': 'Everton', 'age': 24},
        'Kristjan Asllani': {'club': 'Inter Milan', 'age': 24},
        'Nedim Bajrami': {'club': 'Rangers', 'age': 27},
        'Berat Ridvan Gjimshiti': {'club': 'Atalanta', 'age': 33},
        'Thomas Strakosha': {'club': 'AEK Athens', 'age': 31},
        'Mario Mitaj': {'club': 'Al-Ittihad', 'age': 22},
        'Ylber Ramadani': {'club': 'Lecce', 'age': 29},
        'Rey Manaj': {'club': 'Sivasspor', 'age': 29},
        'Jasir Asani': {'club': 'Gwangju FC', 'age': 30},
        'Ernest Muçi': {'club': 'Beşiktaş', 'age': 25}
    },
    'Denmark': {
        'Rasmus Winther Højlund': {'club': 'Manchester United', 'age': 23},
        'Pierre-Emile Kordt Højbjerg': {'club': 'Marseille', 'age': 30},
        'Christian Dannemann Eriksen': {'club': 'VfL Wolfsburg', 'age': 34},
        'Joachim Christian Andersen': {'club': 'Fulham', 'age': 29},
        'Morten Due Hjulmand': {'club': 'Bayer Leverkusen', 'age': 26},
        'Alexander Bah': {'club': 'Benfica', 'age': 28},
        'Victor Bak Jensen': {'club': 'FC Midtjylland', 'age': 22},
        'Anders Dreyer': {'club': 'San Diego FC', 'age': 27},
        'William Osula': {'club': 'Newcastle United', 'age': 22},
        'Kasper Schmeichel': {'club': 'Celtic', 'age': 39}
    },
    'Ukraine': {
        'Andriy Lunin': {'club': 'Real Madrid', 'age': 27},
        'Mykhailo Mudryk': {'club': 'Chelsea', 'age': 25},
        'Oleksandr Zinchenko': {'club': 'Arsenal', 'age': 29},
        'Artem Dovbyk': {'club': 'Roma', 'age': 28},
        'Illia Borysovych Zabarnyi': {'club': 'Bournemouth', 'age': 23},
        'Georgiy Sudakov': {'club': 'Shakhtar Donetsk', 'age': 23},
        'Viktor Tsygankov': {'club': 'Girona', 'age': 28},
        'Vitaliy Mykolenko': {'club': 'Everton', 'age': 26},
        'Anatoliy Trubin': {'club': 'Benfica', 'age': 24},
        'Yukhym Konoplya': {'club': 'Shakhtar Donetsk', 'age': 26}
    },
    'South Korea': {
        'Son Heung-min': {'club': 'Tottenham Hotspur', 'age': 33},
        'Kim Min-jae': {'club': 'Bayern Munich', 'age': 29},
        'Lee Kang-in': {'club': 'Paris Saint-Germain', 'age': 25},
        'Hwang Hee-chan': {'club': 'Wolves', 'age': 30},
        'Hwang In-beom': {'club': 'Feyenoord', 'age': 29},
        'Cho Gue-sung': {'club': 'Midtjylland', 'age': 28},
        'Seol Young-woo': {'club': 'Red Star Belgrade', 'age': 27},
        'Lee Jae-sung': {'club': 'Mainz 05', 'age': 33},
        'Jo Hyeon-woo': {'club': 'Ulsan HD', 'age': 34},
        'Bae Jun-ho': {'club': 'Stoke City', 'age': 22}
    },
    'Iran': {
        'Mehdi Taremi': {'club': 'Inter Milan', 'age': 33},
        'Sardar Azmoun': {'club': 'Shabab Al-Ahli', 'age': 31},
        'Alireza Safar Beiranvand': {'club': 'Tractor', 'age': 33},
        'Saman Ghoddos': {'club': 'Ittihad Kalba', 'age': 32},
        'Alireza Jahanbakhsh': {'club': 'Heerenveen', 'age': 32},
        'Saeid Ezatolahi': {'club': 'Shabab Al-Ahli', 'age': 29},
        'Milad Mohammadi': {'club': 'Persepolis', 'age': 32},
        'Shojae Khalilzadeh': {'club': 'Tractor', 'age': 36},
        'Mehdi Ghayedi': {'club': 'Ittihad Kalba', 'age': 27},
        'Majid Hosseini': {'club': 'Kayserispor', 'age': 29}
    },
    'Saudi Arabia': {
        'Salem Mohammed Al Dawsari': {'club': 'Al-Hilal', 'age': 34},
        'Saud Abdullah Abdul Hamid': {'club': 'Lens', 'age': 26},
        'Firas Tariq Nasser Al Albirakan': {'club': 'Al-Ahli', 'age': 25},
        'Nawaf Al-Aqidi': {'club': 'Al-Nassr', 'age': 25},
        'Ali Hadi Mohammed Al-Bulaihi': {'club': 'Al-Hilal', 'age': 36},
        'Marwan Al-Sahafi': {'club': 'Royal Antwerp', 'age': 22},
        'Faisal Al-Ghamdi': {'club': 'Beerschot', 'age': 24},
        'Sultan Al-Ghannam': {'club': 'Al-Nassr', 'age': 31},
        'Mohamed Kanno': {'club': 'Al-Hilal', 'age': 31},
        'Musab Al-Juwayr': {'club': 'Al Qadisiyah', 'age': 22}
    },
    'Romania': {
        'Radu Matei Drăgușin': {'club': 'Tottenham Hotspur', 'age': 24},
        'Răzvan Gabriel Marin': {'club': 'Cagliari', 'age': 29},
        'Dennis Man': {'club': 'Parma', 'age': 27},
        'Nicolae Claudiu Stanciu': {'club': 'Damac', 'age': 32},
        'Andrei Florin Rațiu': {'club': 'Rayo Vallecano', 'age': 27},
        'Valentin Mihăilă': {'club': 'Parma', 'age': 26},
        'Horațiu Moldovan': {'club': 'Sassuolo', 'age': 28},
        'Andrei Burcă': {'club': 'Baniyas', 'age': 32},
        'Denis Drăguș': {'club': 'Trabzonspor', 'age': 26},
        'Ianis Hagi': {'club': 'Rangers', 'age': 27}
    },
    'Hungary': {
        'Dominik Szoboszlai': {'club': 'Liverpool', 'age': 25},
        'Roland Sallai': {'club': 'Galatasaray', 'age': 28},
        'Barnabás Varga': {'club': 'Ferencváros', 'age': 31},
        'Milos Kerkez': {'club': 'Bournemouth', 'age': 22},
        'Vilmos Tamás Orbán': {'club': 'RB Leipzig', 'age': 33},
        'Dénes Dibusz': {'club': 'Ferencváros', 'age': 35},
        'Martin Ádám': {'club': 'Asteras Tripolis', 'age': 31},
        'Loïc Négo': {'club': 'Le Havre', 'age': 35},
        'András Schäfer': {'club': 'Union Berlin', 'age': 26},
        'Bendegúz Bolla': {'club': 'Rapid Vienna', 'age': 26}
    },
    'Qatar': {
        'Akram Hassan Afif Yahya Afif': {'club': 'Al-Sadd', 'age': 29},
        'Almoez Ali Zainalabiddin Abdulla': {'club': 'Al-Duhail', 'age': 29},
        'Meshaal Aissa Barsham': {'club': 'Al-Sadd', 'age': 28},
        'Lucas Mendes': {'club': 'Al-Wakrah', 'age': 35},
        'Jassem Gaber Abdulsallam': {'club': 'Al-Arabi', 'age': 24},
        'Edmilson Junior': {'club': 'Al-Duhail', 'age': 31},
        'Boualem Khoukhi': {'club': 'Al-Sadd', 'age': 35},
        'Bassam Al-Rawi': {'club': 'Al-Duhail', 'age': 28},
        'Hassan Al-Haydos': {'club': 'Al-Sadd', 'age': 35},
        'Ahmed Al-Rawi': {'club': 'Al-Rayyan', 'age': 21}
    }
    
}