"""
2026 World Cup predicted rosters (as of February 15, 2026).
Sources: Public football databases, transfer news.

Key notes:
  - Mikel Merino (Spain/Arsenal): Injury doubt - foot surgery Feb 9, 2026
  - Christian Pulisic (USA/AC Milan): Career-best season
  - Luka Modrić (Croatia/AC Milan): Final season, contract expires June 2026
  - Lucas Paquetá (Brazil/Flamengo): Moved from West Ham on Jan 30, 2026
"""

rosters_2026 = {
    'Spain': {
        'Unai Simón': 'Athletic Club',
        'David Raya': 'Arsenal',
        'Dani Carvajal': 'Real Madrid',
        'Aymeric Laporte': 'Al-Nassr',
        'Robin Le Normand': 'Atlético Madrid',
        'Dani Vivian': 'Athletic Club',
        'Marc Cucurella': 'Chelsea',
        'Alejandro Grimaldo': 'Bayer Leverkusen',
        'Rodri': 'Manchester City',
        'Fabián Ruiz': 'Paris Saint-Germain',
        'Pedri': 'Barcelona',
        'Gavi': 'Barcelona',
        'Dani Olmo': 'Barcelona',
        'Mikel Merino': 'Arsenal',  # INJURY DOUBT - foot surgery Feb 9
        'Martín Zubimendi': 'Real Sociedad',
        'Álvaro Morata': 'AC Milan',
        'Nico Williams': 'Athletic Club',
        'Lamine Yamal': 'Barcelona',
        'Ferran Torres': 'Barcelona',
        'Mikel Oyarzabal': 'Real Sociedad',
        'Joselu': 'Real Madrid',
        'Ayoze Pérez': 'Real Betis',
        'Bryan Gil': 'Sevilla',
        'Pau Cubarsí': 'Barcelona'
    },
    
    'Argentina': {
        'Emiliano Martínez': 'Aston Villa',
        'Cristian Romero': 'Tottenham Hotspur',
        'Lisandro Martínez': 'Manchester United',
        'Nicolás Otamendi': 'Benfica',
        'Gonzalo Montiel': 'Sevilla',
        'Nahuel Molina': 'Atlético Madrid',
        'Marcos Acuña': 'Sevilla',
        'Nicolás Tagliafico': 'Lyon',
        'Rodrigo De Paul': 'Atlético Madrid',
        'Leandro Paredes': 'Roma',
        'Alexis Mac Allister': 'Liverpool',
        'Enzo Fernández': 'Chelsea',
        'Giovani Lo Celso': 'Tottenham Hotspur',
        'Exequiel Palacios': 'Bayer Leverkusen',
        'Lionel Messi': 'Inter Miami',
        'Julián Álvarez': 'Atlético Madrid',
        'Lautaro Martínez': 'Inter Milan',
        'Nicolás González': 'Fiorentina',
        'Paulo Dybala': 'Roma',
        'Alejandro Garnacho': 'Manchester United',
        'Thiago Almada': 'Botafogo',
        'Valentín Carboni': 'Monza',
        'Franco Mastantuono': 'Real Madrid', # 18yo sensation, moved from River Plate.
        'Nico Paz': 'Como/Argentina',
    },
    
    'England': {
        'Jordan Pickford': 'Everton',
        'Aaron Ramsdale': 'Arsenal',
        'Kyle Walker': 'Manchester City',
        'John Stones': 'Manchester City',
        'Harry Maguire': 'Manchester United',
        'Marc Guéhi': 'Crystal Palace',
        'Luke Shaw': 'Manchester United',
        'Kieran Trippier': 'Newcastle United',
        'Declan Rice': 'Arsenal',
        'Jude Bellingham': 'Real Madrid',
        'Phil Foden': 'Manchester City',
        'Bukayo Saka': 'Arsenal',
        'James Maddison': 'Tottenham Hotspur',
        'Kalvin Phillips': 'Manchester City',
        'Conor Gallagher': 'Tottenham Hotspur',
        'Cole Palmer': 'Chelsea',
        'Harry Kane': 'Bayern Munich',
        'Marcus Rashford': 'Manchester United',
        'Raheem Sterling': 'Chelsea',
        'Ollie Watkins': 'Aston Villa',
        'Ivan Toney': 'Brentford',
        'Jarrod Bowen': 'West Ham United',
        'Jack Grealish': 'Manchester City',
    },
    
    'France': {
        'Mike Maignan': 'AC Milan',
        'Alphonse Areola': 'West Ham United',
        'Jules Koundé': 'Barcelona',
        'Dayot Upamecano': 'Bayern Munich',
        'Ibrahima Konaté': 'Liverpool',
        'William Saliba': 'Arsenal',
        'Theo Hernández': 'AC Milan',
        'Benjamin Pavard': 'Inter Milan',
        'Aurélien Tchouaméni': 'Real Madrid',
        'Eduardo Camavinga': 'Real Madrid',
        'N\'Golo Kanté': 'Al-Ittihad',
        'Adrien Rabiot': 'Marseille',
        'Antoine Griezmann': 'Atlético Madrid',
        'Ousmane Dembélé': 'Paris Saint-Germain',
        'Kylian Mbappé': 'Real Madrid',
        'Randal Kolo Muani': 'Paris Saint-Germain',
        'Marcus Thuram': 'Inter Milan',
        'Kingsley Coman': 'Bayern Munich',
        'Bradley Barcola': 'Paris Saint-Germain',
        'Christopher Nkunku': 'Chelsea',
        'Youssouf Fofana': 'AC Milan',
        'Warren Zaïre-Emery': 'Paris Saint-Germain',
    },
    
    'Germany': {
        'Marc-André ter Stegen': 'Barcelona',
        'Oliver Baumann': 'TSG Hoffenheim',
        'Antonio Rüdiger': 'Real Madrid',
        'Nico Schlotterbeck': 'Borussia Dortmund',
        'Jonathan Tah': 'Bayer Leverkusen',
        'David Raum': 'RB Leipzig',
        'Joshua Kimmich': 'Bayern Munich',
        'Jamal Musiala': 'Bayern Munich',
        'Florian Wirtz': 'Bayer Leverkusen',
        'Leon Goretzka': 'Bayern Munich',
        'Kai Havertz': 'Arsenal',
        'Leroy Sané': 'Bayern Munich',
        'Serge Gnabry': 'Bayern Munich',
        'Niclas Füllkrug': 'AC Milan',
        'Karim Adeyemi': 'Borussia Dortmund',
        'Pascal Groß': 'Borussia Dortmund',
        'Robert Andrich': 'Bayer Leverkusen',
        'Maximilian Mittelstädt': 'VfB Stuttgart',
        'Aleksandar Pavlović': 'Bayern Munich',
        'Chris Führich': 'VfB Stuttgart',
        'Emre Can': 'Borussia Dortmund',
        'Julian Brandt': 'Borussia Dortmund',
        'Lennart Karl': 'Bayern Munich'
    },
    
    'Brazil': {
        'Alisson': 'Liverpool',
        'Ederson': 'Manchester City',
        'Danilo': 'Flamengo',
        'Marquinhos': 'Paris Saint-Germain',
        'Éder Militão': 'Real Madrid',
        'Gabriel Magalhães': 'Arsenal',
        'Alex Sandro': 'Flamengo',
        'Renan Lodi': 'Al-Hilal',
        'Casemiro': 'Manchester United',
        'Fabinho': 'Al-Ittihad',
        'Bruno Guimarães': 'Newcastle United',
        'Lucas Paquetá': 'Flamengo',  # CORRECTED - moved Jan 30, 2026
        'Neymar': 'Al-Hilal',
        'Vinícius José Paixão de Oliveira Júnior': 'Real Madrid',
        'Rodrygo': 'Real Madrid',
        'Raphinha': 'Barcelona',
        'Antony': 'Manchester United',
        'Gabriel Jesus': 'Arsenal',
        'Richarlison': 'Tottenham Hotspur',
        'Gabriel Martinelli': 'Arsenal',
        'Pedro': 'Flamengo',
        'Endrick': 'Real Madrid',
        'Savinho': 'Manchester City',
        'Estêvão': 'Bayern Munich',         # Moved from Chelsea/Palmeiras. 
        'Andrey Santos': 'Chelsea'
    },
    
    'Netherlands': {
        'Bart Verbruggen': 'Brighton & Hove Albion',
        'Virgil van Dijk': 'Liverpool',
        'Matthijs de Ligt': 'Manchester United',
        'Nathan Aké': 'Manchester City',
        'Denzel Dumfries': 'Inter Milan',
        'Jeremie Frimpong': 'Bayer Leverkusen',
        'Frenkie de Jong': 'Barcelona',
        'Tijjani Reijnders': 'AC Milan',
        'Ryan Gravenberch': 'Liverpool',
        'Georginio Wijnaldum': 'Al-Ettifaq',
        'Xavi Simons': 'RB Leipzig',
        'Teun Koopmeiners': 'Juventus',
        'Memphis Depay': 'Corinthians',
        'Cody Gakpo': 'Liverpool',
        'Donyell Malen': 'Borussia Dortmund',
        'Steven Bergwijn': 'Al-Ittihad',
        'Wout Weghorst': 'Ajax',
        'Brian Brobbey': 'Ajax',
        'Joey Veerman': 'PSV Eindhoven',
        'Mats Wieffer': 'Brighton & Hove Albion',
        'Lutsharel Geertruida': 'RB Leipzig',
        'Jurriën Timber': 'Arsenal',
        'Ian Maatsen': 'Aston Villa',
    },
    
    'Portugal': {
        'Diogo Costa': 'Porto',
        'Rui Patrício': 'Atalanta',
        'Rúben Dias': 'Manchester City',
        'Gonçalo Inácio': 'Sporting CP',
        'Danilo Pereira': 'Paris Saint-Germain',
        'João Cancelo': 'Al-Hilal',
        'Nuno Mendes': 'Paris Saint-Germain',
        'João Palhinha': 'Bayern Munich',
        'Rúben Neves': 'Al-Hilal',
        'Bernardo Silva': 'Manchester City',
        'Bruno Fernandes': 'Manchester United',
        'Vitinha': 'Paris Saint-Germain',
        'João Félix': 'Chelsea',
        'Rafael Leão': 'AC Milan',
        'Cristiano Ronaldo': 'Al-Nassr',
        'Gonçalo Ramos': 'Paris Saint-Germain',
        'Diogo Jota': 'Liverpool',
        'Pedro Neto': 'Chelsea',
        'Otávio': 'Al-Nassr',
        'Matheus Nunes': 'Manchester City',
        'Francisco Conceição': 'Juventus',
        'João Neves': 'Paris Saint-Germain',
        'António Silva': 'Benfica',
    },
    
    'USA': {
        'Matt Turner': 'Crystal Palace',
        'Zack Steffen': 'Colorado Rapids',
        'Sergiño Dest': 'PSV Eindhoven',
        'Antonee Robinson': 'Fulham',
        'Chris Richards': 'Crystal Palace',
        'Cameron Carter-Vickers': 'Celtic',
        'Tim Ream': 'Charlotte FC',
        'Joe Scally': 'Borussia Mönchengladbach',
        'Tyler Adams': 'Bournemouth',
        'Weston McKennie': 'Juventus',
        'Yunus Musah': 'Atalanta',
        'Gio Reyna': 'Borussia Dortmund',
        'Luca de la Torre': 'Celta Vigo',
        'Malik Tillman': 'PSV Eindhoven',
        'Christian Pulisic': 'AC Milan',  # FORM NOTE: Career-best season
        'Timothy Weah': 'Juventus',
        'Folarin Balogun': 'Monaco',
        'Ricardo Pepi': 'PSV Eindhoven',
        'Haji Wright': 'Coventry City',
        'Josh Sargent': 'Norwich City',
        'Brenden Aaronson': 'Leeds United',
        'Johnny Cardoso': 'Real Betis',
        'Tanner Tessmann': 'Inter Milan',
    },

    'Uruguay': {
        'Sergio Rochet': 'Internacional',
        'Federico Valverde': 'Real Madrid', 
        'Darwin Núñez': 'Liverpool',
        'Ronald Araújo': 'Barcelona',
        'Manuel Ugarte': 'Manchester United',
        'José María Giménez': 'Atlético Madrid',
        'Mathías Olivera': 'Napoli',
        'Facundo Pellistri': 'Panathinaikos',
        'Nicolás de la Cruz': 'Flamengo',
        'Nahitan Nández': 'Al-Qadsiah',
        'Luciano Rodríguez': 'Bahia', # UPDATED: The 2026 breakout 'Lucho'.
    },

    'Croatia': {
        'Dominik Livaković': 'Fenerbahçe',
        'Joško Gvardiol': 'Manchester City',
        'Dejan Lovren': 'PAOK',
        'Domagoj Vida': 'AEK Athens',
        'Josip Juranović': 'Union Berlin',
        'Borna Sosa': 'Torino',
        'Luka Modrić': 'AC Milan',  # CORRECTED - Moved July 2025, contract through June 2026
        'Mateo Kovačić': 'Manchester City',
        'Marcelo Brozović': 'Al-Nassr',
        'Mario Pašalić': 'Atalanta',
        'Luka Sučić': 'Red Bull Salzburg',
        'Lovro Majer': 'Wolfsburg',
        'Ivan Perišić': 'PSV Eindhoven',
        'Andrej Kramarić': 'TSG Hoffenheim',
        'Bruno Petković': 'Dinamo Zagreb',
        'Marko Pjaca': 'Dinamo Zagreb',
        'Nikola Vlašić': 'Torino',
        'Ante Budimir': 'Osasuna',
        'Mislav Oršić': 'Trabzonspor',
        'Josip Stanišić': 'Bayern Munich',
        'Martin Erlić': 'Sassuolo',
        'Joško Šutalo': 'Hellas Verona',
        'Kristijan Jakić': 'Augsburg',
    },
    

    'Italy': {
        'Gianluigi Donnarumma': 'Paris Saint-Germain',
        'Alessandro Bastoni': 'Inter Milan',
        'Riccardo Calafiori': 'Arsenal',
        'Nicolò Barella': 'Inter Milan',
        'Sandro Tonali': 'Newcastle United',
        'Mateo Retegui': 'Atalanta',
        'Federico Chiesa': 'Liverpool',
        'Daniel Maldini': 'Monza', 
        'Destiny Udogie': 'Tottenham Hotspur',
        'Gianluca Scamacca': 'Atalanta',
        'Giorgio Scalvini': 'Atalanta',
    },

    'Mexico': {
        'Luis Malagón': 'Club América',
        'Edson Álvarez': 'West Ham United',
        'Santiago Giménez': 'AC Milan', 
        'Hirving Lozano': 'San Diego FC', 
        'Raúl Jiménez': 'Fulham',
        'Gilberto Mora': 'Feyenoord', # UPDATED: Moved to Europe Jan 2026.
        'Johan Vásquez': 'Genoa',
        'Jorge Sánchez': 'Cruz Azul',
        'César Montes': 'Lokomotiv Moscow',
    },

    'Belgium': {
        'Thibaut Courtois': 'Real Madrid', 
        'Kevin De Bruyne': 'Napoli', 
        'Jérémy Doku': 'Manchester City',
        'Amadou Onana': 'Aston Villa',
        'Loïs Openda': 'Juventus', 
        'Charles De Ketelaere': 'Atalanta',
        'Youri Tielemans': 'Aston Villa', 
        'Arthur Theate': 'Eintracht Frankfurt',
        'Maxim De Cuyper': 'Club Brugge',
    },

    'Japan': {
        'Zion Suzuki': 'Parma',
        'Wataru Endo': 'Liverpool',
        'Takefusa Kubo': 'Real Sociedad',
        'Kaoru Mitoma': 'Brighton',
        'Ko Itakura': 'Borussia Mönchengladbach',
        'Hiroki Ito': 'Bayern Munich',
        'Ritsu Doan': 'SC Freiburg',
        'Ayase Ueda': 'Feyenoord',
        'Takehiro Tomiyasu': 'Arsenal',
        'Daichi Kamada': 'Crystal Palace',
        'Keito Nakamura': 'Reims',
    },

    'Morocco': {
        'Yassine Bounou': 'Al-Hilal', 'Achraf Hakimi': 'Paris Saint-Germain', 'Nayef Aguerd': 'Real Sociedad', 
        'Noussair Mazraoui': 'Manchester United', 'Brahim Abdelkader Díaz': 'Real Madrid', 'Sofyan Amrabat': 'Fenerbahçe',
        'Azzedine Ounahi': 'Panathinaikos', 'Hakim Ziyech': 'Galatasaray', 'Youssef En-Nesyri': 'Fenerbahçe'
    },
    'Colombia': {
        'Camilo Vargas': 'Atlas', 'Daniel Muñoz': 'Crystal Palace', 'Davinson Sánchez': 'Galatasaray',
        'Jhon Lucumí': 'Bologna', 'Richard Ríos': 'Palmeiras', 'James Rodríguez': 'Rayo Vallecano',
        'Luis Fernando Díaz Marulanda': 'Bayern München', 'Jhon Durán': 'Aston Villa', 'Luis Sinisterra': 'Bournemouth'
    },
    'Switzerland': {
        'Gregor Kobel': 'Borussia Dortmund', 'Yann Sommer': 'Internazionale', 'Manuel Akanji': 'Manchester City',
        'Granit Xhaka': 'Bayer Leverkusen', 'Denis Zakaria': 'Monaco', 'Remo Freuler': 'Bologna',
        'Dan Ndoye': 'Bologna', 'Ruben Vargas': 'Sevilla', 'Zeki Amdouni': 'Benfica'
    },
    'Denmark': {
        'Kasper Schmeichel': 'Celtic', 'Joachim Andersen': 'Fulham', 'Andreas Christensen': 'Barcelona',
        'Patrick Dorgu': 'Manchester United', 'Morten Hjulmand': 'Sporting CP', 'Pierre-Emile Højbjerg': 'Marseille',
        'Christian Eriksen': 'VfL Wolfsburg', 'Rasmus Højlund': 'Napoli', 'Mikkel Damsgaard': 'Brentford'
    },
    'South Korea': {
        'Jo Hyeon-woo': 'Ulsan HD', 'Kim Min-jae': 'Bayern München', 'Lee Kang-in': 'Paris Saint-Germain',
        'Hwang Hee-chan': 'Wolverhampton Wanderers', 'Heung-Min Son': 'LAFC', 'Seol Young-woo': 'Crvena Zvezda',
        'Bae Jun-ho': 'Stoke City'
    },
    'Ecuador': {
        'Hernán Galíndez': 'Huracán', 'Piero Hincapié': 'Arsenal', 'Willian Pacho': 'Paris Saint-Germain',
        'Pervis Estupiñán': 'Brighton & Hove Albion', 'Moisés Caicedo': 'Chelsea', 'Kendry Páez': 'Chelsea',
        'Enner Valencia': 'Pachuca', 'Gonzalo Plata': 'Flamengo', 'Jeremy Sarmiento': 'Brighton & Hove Albion'
    },
    'Turkey': {
        'Altay Bayındır': 'Manchester United', 'Merih Demiral': 'Al-Ahli', 'Ferdi Kadıoğlu': 'Brighton & Hove Albion',
        'Hakan Çalhanoğlu': 'Internazionale', 'Arda Güler': 'Real Madrid', 'Kenan Yıldız': 'Juventus',
        'Can Uzun': 'Eintracht Frankfurt', 'Kerem Aktürkoğlu': 'Benfica', 'Barış Alper Yılmaz': 'Galatasaray'
    },
    'Senegal': {
        'Édouard Mendy': 'Al-Ahli', 'Kalidou Koulibaly': 'Al-Hilal', 'Pape Matar Sarr': 'Tottenham Hotspur',
        'Nicolas Jackson': 'Chelsea', 'Sadio Mané': 'Al-Nassr', 'Ismaïla Sarr': 'Crystal Palace',
        'Idrissa Gana Gueye': 'Everton', 'Lamine Camara': 'Monaco'
    },
    'Canada': {
        'Maxime Crépeau': 'Portland Timbers', 
        'Alphonso Davies': 'Bayern München',  # CORRECTED: Stayed at Bayern
        'Moïse Bombito': 'Nice',
        'Ismaël Koné': 'Marseille', 
        'Stephen Eustáquio': 'Los Angeles FC', # UPDATE: Joined Son at LAFC
        'Jonathan David': 'Juventus',          # CORRECTED: Move to Juve, not Villa
        'Jacob Shaffelburg': 'Los Angeles FC', # UPDATE: High-profile MLS move
        'Tajon Buchanan': 'Internazionale'
    
    },
    'Nigeria': {
        'Stanley Nwabali': 'Chippa United', 'Calvin Bassey': 'Fulham', 'William Troost-Ekong': 'Al-Kholood',
        'Wilfred Ndidi': 'Beşiktaş', 'Alex Iwobi': 'Fulham', 'Ademola Lookman': 'Atalanta',
        'VVictor James Osimhen': 'Galatasaray', 'Victor Boniface': 'Bayer Leverkusen', 'Samuel Chukwueze': 'AC Milan'
    },
    'Serbia': {
        'Vanja Milinković-Savić': 'Torino', 'Nikola Milenković': 'Nottingham Forest', 'Strahinja Pavlović': 'AC Milan',
        'Dušan Tadić': 'Fenerbahçe', 'Lazar Samardžić': 'Atalanta', 'Aleksandar Mitrović': 'Al-Hilal',
        'Dušan Vlahović': 'Juventus', 'Filip Kostić': 'Fenerbahçe'
    },
    'Austria': {
        'Patrick Pentz': 'Brøndby', 'David Alaba': 'Real Madrid', 'Kevin Danso': 'Tottenham Hotspur',
        'Konrad Laimer': 'Bayern München', 'Marcel Sabitzer': 'Borussia Dortmund', 'Christoph Baumgartner': 'RB Leipzig',
        'Nicolas Seiwald': 'RB Leipzig', 'Marko Arnautović': 'Crvena Zvezda'
    },
    'Poland': {
        'Marcin Bułka': 'Nice', 'Jakub Kiwior': 'FC Porto', 'Matty Cash': 'Aston Villa',
        'Piotr Zieliński': 'Internazionale', 'Sebastian Szymański': 'Fenerbahçe', 'Nicola Zalewski': 'Roma',
        'Robert Lewandowski': 'Barcelona', 'Karol Świderski': 'Charlotte FC'
    },
    'Ghana': {
        'Lawrence Ati-Zigi': 'St. Gallen', 'Mohammed Salisu': 'Monaco', 'Tariq Lamptey': 'Brighton & Hove Albion',
        'Thomas Partey': 'Arsenal', 'Mohammed Kudus': 'West Ham United', 'Antoine Semenyo': 'Bournemouth',
        'Iñaki Williams': 'Athletic Club', 'Jordan Ayew': 'Leicester City'
    }
}

MANAGER_TENURE = {
    # --- LONG-TERM STABILITY (High Bonus) ---
    'France': 2012,      # Didier Deschamps
    'Croatia': 2017,     # Zlatko Dalić
    'Argentina': 2018,   # Lionel Scaloni
    'Japan': 2018,       # Hajime Moriyasu
    'Spain': 2022,       # Luis de la Fuente
    'Morocco': 2022,     # Walid Regragui
    'Colombia': 2022,    # Néstor Lorenzo
    'Austria': 2022,     # Ralf Rangnick
    'Serbia': 2021,      # Dragan Stojković

    # --- MID-TERM / ESTABLISHED (Standard Bonus) ---
    'Germany': 2023,     # Julian Nagelsmann
    'Portugal': 2023,    # Roberto Martínez
    'Netherlands': 2023, # Ronald Koeman
    'Uruguay': 2023,     # Marcelo Bielsa
    'Turkey': 2023,      # Vincenzo Montella

    # --- NEW ERA / 2026 CYCLE APPOINTMENTS (Stability Penalty) ---
    'USA': 2024,         # Mauricio Pochettino
    'Canada': 2024,      # Jesse Marsch
    'Mexico': 2024,      # Javier Aguirre
    'Brazil': 2025,      # Carlo Ancelotti (Started May 2025)
    'England': 2025,     # Thomas Tuchel (Started Jan 2025)
    'Italy': 2025,       # Gennaro Gattuso (Started June 2025)
    'Belgium': 2025,     # Rudi Garcia
    'Poland': 2025,      # Jan Urban
    'Nigeria': 2025,     # Éric Chelle
    'Denmark': 2024,     # Brian Riemer
    'South Korea': 2024, # Hong Myung-bo
    'Ecuador': 2024,     # Sebastián Beccacece
    'Senegal': 2024,     # Pape Thiaw
    'Ghana': 2024,       # Otto Addo
    'Serbia': 2025
}