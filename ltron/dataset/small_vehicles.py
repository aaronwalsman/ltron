#!/usr/bin/env python
import ltron.settings as settings
from ltron.dataset.build_dataset import build_dataset

investigate = ''' 
10036-1 - Pizza To Go.mpd
10128-1 - Train Level Crossing.mpd
10156-1 - LEGO Truck.mpd
10159-1 - City Airport -City Logo Box.mpd
10184 - Town Plan.mpd
10197-1 - Fire Brigade.mpd
10213-1 - Shuttle Adventure.mpd
10214-1 - Tower Bridge.mpd
10264 - Corner Garage.mpd
1029-1 - Milk Delivery Truck - Tine.mpd
106-1 - UNICEF Van.mpd
107-2 - Canada Post Mail Truck.mpd
1096-1 - Race Buggy.mpd
1097-1 - Res-Q Runner.mpd
1180-1 - Space Port Moon Buggy.mpd
1462-1 - Galactic Scout.mpd
1472-1 - Holiday Home.mpd
=============== 2021/2/22 below here ======================
1477-1 - %7BRed Race Car Number 3%7D.mpd
1478-1 - Mobile Satellite Up-Link.mpd
1479-1 - 2-Pilot Craft.mpd
1484-1 - Weetabix Town House.mpd
1489-1 - Mobile Car Crane.mpd
1490-1 - Town Bank.mpd
1496-1 - Rally Car.mpd
1557-1 - Scooter.mpd
1591 - Danone Truck.mpd
1620-1 - Astro Dart.mpd
1621-1 - Lunar MPV Vehicle.mpd
1631-1 - Black Racer.mpd
1632-1 - Motor Boat.mpd
1633-1 - Loader Tractor.mpd
1656-1 - Evacuation Team.mpd
1682-1 - Space Shuttle.mpd
1702-1 - Fire Fighter 4 x 4.mpd
1704-1 - Ice Planet Satellite Plough.mpd
1713-1 - Shipwrecked Pirate.mpd
1714-1 - Surveillance Scooter.mpd
1731-1 - Ice Planet Scooter.mpd
1772-1 - Airport Container Truck.mpd
1773-1 - Airline Main_YbM0Kh4.mpd
1775-1 - Jet.mpd
1792-1 - Pleasure Cruiser.mpd
1793-1 - Space Station Zenon.mpd
1808-1 - Light Aircraft and _3j8215M.mpd
1831-1 - Maersk Line Container Lorry.mpd
1875-1 - Meteor Monitor.mpd
1887-1 - Scout Patrol Ship.mpd
1896-1 - Trauma Team.mpd
1916-1 - Starion Patrol.mpd
1952-1 - Dairy Tanker.mpd
1969-1 - Mini Robot.mpd
1974-4 - Star Quest.mpd
=============== 2021/2/22 above here ======================
20006-1 - Clone Turbo Tank - Mini.mpd
20009-1 - AT-TE Walker - Mini.mpd
20011-1 - Garbage Truck.mpd
20014-1 - 4 x 4 Dynamo.mpd
20019-1 - Slave I.mpd
20021-1 - Bounty Hunter Gunship - Mini.mpd
2140-1 - ANWB Roadside Assistance Crew.mpd
2148-1 - LEGO Truck.mpd
2149-1 - Color Line Container Lorry.mpd
2150-1 - Train Station.mpd
2531-1 - Rescue Helicopter and Jeep.mpd
2541-1 - Adventurers Car.mpd
2542-1 - Adventurers Aeroplane.mpd
2584-1 - Biker Bob.mpd
30030-1 - Racing Car.mpd
30033-1 - Racing Truck.mpd
30034-1 - Racing Tow Truck.mpd
30035-1 - Racing Car.mpd
30036-1 - Buggy Racer.mpd
30050-1 - Republic Attack Shuttle - Mini.mpd
30051-1 - X-wing Fighter - Mini.mpd
30052-1 - AAT - Mini.mpd
30053-1 - Republic Attack Cruiser - Mini.mpd
============= Stopped here for the night =======================
'''

other_intermediate = '''
1972 - Go Kart.mpd
'''

large = '''
10242 - Mini Cooper.mpd
10248 - Ferrari F40.mpd
10252-1 - Volkswagen Beetle.mpd
10258-1 - London Bus.mpd
10271 - Fiat 500.mpd
21307-1 - Caterham Seven 620R.mpd
21307 - Caterham Seven 620R.mpd

'''

dataset_paths = [
    '10036-1 - Pizza To Go.mpd:10036 - car.ldr',
    '10128-1 - Train Level Crossing.mpd:10128 - car.ldr',
    '10156-1 - LEGO Truck.mpd:10156 - truck.ldr',
    # big and has custom parts
    #'10159-1 - City Airport -City Logo Box.mpd:10159 - airplane.ldr',
    '10159-1 - City Airport -City Logo Box.mpd:10159 - helicopter.ldr',
    '10159-1 - City Airport -City Logo Box.mpd:10159 - baggage car.ldr',
    '10159-1 - City Airport -City Logo Box.mpd:10159 - baggage trailer.ldr',
    '10184 - Town Plan.mpd:10184 - car.ldr',
    '10184 - Town Plan.mpd:10184 - truck.ldr',
    # stuff is poorly named here and model is weird, come back if desperate
    #10197-1 - Fire Brigade.mpd
    '10213-1 - Shuttle Adventure.mpd:10213 - car.ldr',
    '10214-1 - Tower Bridge.mpd:10214 - sub-model a - black london taxi.ldr',
    '10214-1 - Tower Bridge.mpd:10214 - sub-model b - green automobile.ldr',
    '10214-1 - Tower Bridge.mpd:10214 - sub-model c - yellow truck.ldr',
    '10214-1 - Tower Bridge.mpd:10214 - sub-model d - red double-decker bus.ldr',
    # poorly named and model is large, come back if desperate
    #'10264 - Corner Garage.mpd'
    '1029-1 - Milk Delivery Truck - Tine.mpd:1029 - car.ldr',
    '106-1 - UNICEF Van.mpd:106 - car.ldr',
    '107-2 - Canada Post Mail Truck.mpd:107-2 - m-1a.ldr',
    '107-2 - Canada Post Mail Truck.mpd:107-2 - m-1b.ldr',
    '1096-1 - Race Buggy.mpd',
    '1097-1 - Res-Q Runner.mpd:1097 - jetski.ldr',
    '1180-1 - Space Port Moon Buggy.mpd',
    '1462-1 - Galactic Scout.mpd:1462 - spaceship.ldr',
    '1472-1 - Holiday Home.mpd:1472 - car 1.ldr',
    '1472-1 - Holiday Home.mpd:1472 - car 2.ldr',
    '1472-1 - Holiday Home.mpd:1472 - boat.ldr',
    '1472-1 - Holiday Home.mpd:1472 - trailer.ldr',
    #=============== 2021/2/22 below here ======================
    '1477-1 - %7BRed Race Car Number 3%7D.mpd',
    '1478-1 - Mobile Satellite Up-Link.mpd',
    '1479-1 - 2-Pilot Craft.mpd:1479 - spaceship.ldr',
    '1484-1 - Weetabix Town House.mpd:1484 - smallcar.ldr',
    '1489-1 - Mobile Car Crane.mpd:1489 - passenger car.ldr',
    '1489-1 - Mobile Car Crane.mpd:1489 - car crane.ldr',
    '1490-1 - Town Bank.mpd:1490 - car.ldr',
    '1496-1 - Rally Car.mpd:1496 - car.ldr',
    '1557-1 - Scooter.mpd:1557 - space scooter - scooter.ldr',
    '1591 - Danone Truck.mpd',
    '1620-1 - Astro Dart.mpd:1620 - spaceship.ldr',
    '1621-1 - Lunar MPV Vehicle.mpd:1621 - vehicle.ldr',
    '1631-1 - Black Racer.mpd',
    '1632-1 - Motor Boat.mpd',
    '1633-1 - Loader Tractor.mpd',
    '1656-1 - Evacuation Team.mpd:1656 - tractor.ldr',
    '1656-1 - Evacuation Team.mpd:1656 - cart.ldr',
    '1656-1 - Evacuation Team.mpd:1656 - car.ldr',
    '1656-1 - Evacuation Team.mpd:1656 - truck.ldr',
    '1656-1 - Evacuation Team.mpd:1656 - trailer.ldr',
    '1682-1 - Space Shuttle.mpd:1682 - car.ldr',
    '1682-1 - Space Shuttle.mpd:1682 - trailer.ldr',
    '1702-1 - Fire Fighter 4 x 4.mpd:1702 - car.ldr',
    '1704-1 - Ice Planet Satellite Plough.mpd:1704 - vehicle + container.ldr',
    '1713-1 - Shipwrecked Pirate.mpd:1713 - raft.ldr',
    '1714-1 - Surveillance Scooter.mpd:1714 - spaceship.ldr',
    '1731-1 - Ice Planet Scooter.mpd:1731 - spaceship.ldr',
    '1772-1 - Airport Container Truck.mpd:1772 - car.ldr',
    '1772-1 - Airport Container Truck.mpd:1772 - container.ldr',
    '1773-1 - Airline Main_YbM0Kh4.mpd:1773 - car.ldr',
    '1773-1 - Airline Main_YbM0Kh4.mpd:1773 - trailer.ldr',
    '1775-1 - Jet.mpd:1775 - airplane.ldr',
    '1792-1 - Pleasure Cruiser.mpd:1792 - boat.ldr',
    '1793-1 - Space Station Zenon.mpd:1793 - vehicle.ldr',
    '1793-1 - Space Station Zenon.mpd:1793 - cockpit.ldr',
    '1808-1 - Light Aircraft and _3j8215M.mpd:1808 - airplane.ldr',
    '1831-1 - Maersk Line Container Lorry.mpd:1831 - truck.ldr',
    '1831-1 - Maersk Line Container Lorry.mpd:1831 - trailer.ldr',
    '1831-1 - Maersk Line Container Lorry.mpd:1831 - container.ldr',
    '1875-1 - Meteor Monitor.mpd:1875 - spaceship.ldr',
    '1887-1 - Scout Patrol Ship.mpd:1887 - spaceship.ldr',
    '1896-1 - Trauma Team.mpd:1896 - car 1.ldr',
    '1896-1 - Trauma Team.mpd:1896 - helicopter.ldr',
    '1896-1 - Trauma Team.mpd:1896 - car 2.ldr',
    '1916-1 - Starion Patrol.mpd:1916 - spaceship.ldr',
    '1952-1 - Dairy Tanker.mpd:1952 - car.ldr',
    '1952-1 - Dairy Tanker.mpd:1952 - trailer.ldr',
    '1969-1 - Mini Robot.mpd:1969 - robot.ldr',
    '1974-4 - Star Quest.mpd:1974 - vehicle.ldr',
    #===================== FINISHED HERE =====================
    #3/8/2021 below here
    '20006-1 - Clone Turbo Tank - Mini.mpd',
    '20009-1 - AT-TE Walker - Mini.mpd',
    '20011-1 - Garbage Truck.mpd',
    '20014-1 - 4 x 4 Dynamo.mpd',
    '20019-1 - Slave I.mpd',
    '20021-1 - Bounty Hunter Gunship - Mini.mpd',
    '2140-1 - ANWB Roadside Assistance Crew.mpd:2140 - car 1.ldr',
    '2140-1 - ANWB Roadside Assistance Crew.mpd:2140 - car 2.ldr',
    '2140-1 - ANWB Roadside Assistance Crew.mpd:2140 - car 3.ldr',
    '2148-1 - LEGO Truck.mpd:2148 - truck.ldr',
    '2149-1 - Color Line Container Lorry.mpd:2149 - truck.ldr',
    '2149-1 - Color Line Container Lorry.mpd:2149 - trailer.ldr',
    '2149-1 - Color Line Container Lorry.mpd:2149 - container.ldr' # not a vhc
    '2150-1 - Train Station.mpd:2150 - luggage car 1.ldr',
    '2150-1 - Train Station.mpd:2150 - luggage car 2.ldr',
    '2531-1 - Rescue Helicopter and Jeep.mpd:2531 - car.ldr',
    '2531-1 - Rescue Helicopter and Jeep.mpd:2531 - helicopter.ldr',
    '2541-1 - Adventurers Car.mpd:2541 - car.ldr',
    '2542-1 - Adventurers Aeroplane.mpd:2542 - aeroplane.ldr',
    
    # from tiny turbos 3
    '30030-1 - Racing Car.mpd',
    '30033-1 - Racing Truck.mpd',
    '30034-1 - Racing Tow Truck.mpd',
    '30035-1 - Racing Car.mpd',
    '30036-1 - Buggy Racer.mpd',
    
    # scanning again
    '30050-1 - Republic Attack Shuttle - Mini.mpd',
    '30051-1 - X-wing Fighter - Mini.mpd',
    '30052-1 - AAT - Mini.mpd',
    '30053-1 - Republic Attack Cruiser - Mini.mpd',
    '30054-1 - AT-ST - Mini.mpd',
    '30055-1 - Vulture Droid - Mini.mpd',
    '30056 - Star Destroyer.mpd',
    '30090-1 - Desert Glider.mpd', # contains minifig
    '30091-1 - Desert Rover.mpd', # contains minifig
    '3015-1 - Space Police Car.mpd:3015 - spaceship.ldr',
    '30161-1 - Batmobile.mpd',
    '30181-1 - Helicopter.ldr',
    '30190 - Ferrari 150deg Italia.mpd',
    '30191-1 - Scuderia Ferrari Truck.mpd',
    '30192-1 - Ferrari F40.mpd',
    #'30193-1 - 250 GT Berlinetta.mpd', # incomplete
    '30194-1 - 458 Italia.mpd',
    '30195-1 - FXX.mpd',
    '30277 - First Order Star Destroyer.mpd',
    #'30283-1 - Off-Road.mpd', # incomplete
    '30284-1 - Tractor.mpd',
    '30300-1 - The Batman Tumbler.mpd',
    '30311-1 - Swamp Police Helicopter.mpd:30311 - helicopter.ldr',
    '30312-1 - Demolition Driller.mpd:30312 - driller.ldr',
    '30313-1 - Garbage Truck.mpd:30313 - truck.ldr',
    '3056-1 - Go-Kart.mpd', # contains minifig
    '30572 - Race Car.mpd',
    '3063-1 - Heartlake Flying Club.mpd:3063 - plane.ldr' # contains friends mfg
    # STOPPED HERE
    
    
    
    # tiny turbos continues here
    '4096 - Micro Wheels - AB Forklift.mpd',
    '4096 - Micro Wheels - AB Loader.mpd',
    '4096 - Micro Wheels - AB Truck and Trailer.mpd',
    '4096 - Micro Wheels - EB Combine Harvester.mpd',
    '4096 - Micro Wheels - EB Crane.mpd',
    '4096 - Micro Wheels - EB Tractor and Trailer.mpd',
    '4096 - Micro Wheels - EB Truck.mpd',
    '4096 - Micro Wheels - QB 4WD.mpd',
    '4096 - Micro Wheels - QB Cement Mixer.mpd',
    '4096 - Micro Wheels - QB Formula1.mpd',
    '4096 - Micro Wheels - QB Roadster 1.mpd',
    '4096 - Micro Wheels - QB Roadster 2.mpd',
    '4096 - Micro Wheels - QB Truck.mpd',
    '4947-1 - Yellow and Black Racer.mpd',
    '4948-1 - Black and Red Racer.mpd',
    '4949-1 - Blue and Yellow Racer.mpd',
    '7611-1 - Police Car.mpd',
    '7612-1 - Muscle Car.mpd',
    '7613-1 - Track Racer.mpd',
    '7800-1 - Off Road Racer.mpd',
    '7801-1 - Red Racer Polybag.mpd',
    '7802-1 - Black Racer Polybag.mpd',
    '8119-1 - Thunder Racer - Combination with 8122.mpd',
    '8119-1 - Thunder Racer.mpd',
    '8120-1 - Rally Sprinter - Combinaton with 8121.mpd',
    '8120-1 - Rally Sprinter.mpd',
    '8121-1 - Track Marshal.mpd',
    '8122-1 - Desert Viper.mpd',
    '8123-1 - Ferrari F1 Racers.mpd:8123 - 1.ldr',
    '8124-1 - Ice Rally.mpd:8124 - 1.ldr',
    '8124-1 - Ice Rally.mpd:8124 - 2.ldr',
    '8125-1 - Thunder Raceway.mpd:8125 - 8125-1.ldr',
    '8125-1 - Thunder Raceway.mpd:8125 - 8125-2.ldr',
    '8126-1 - Desert Challenge.mpd:8126 - 8126-1.ldr',
    '8126-1 - Desert Challenge.mpd:8126 - 8126-2.ldr',
    '8130-1 - Terrain Crusher - Combination with 8131.mpd',
    '8130-1 - Terrain Crusher.mpd',
    '8131-1 - Raceway Rider.mpd',
    '8132-1 - Night Driver - Combination with 8133.mpd',
    '8132-1 - Night Driver.mpd',
    '8133-1 - Rally Rider.mpd',
    '8134-1 - Night Crusher.mpd:8134 - 8134-1.ldr',
    '8134-1 - Night Crusher.mpd:8134 - 8134-2.ldr',
    '8135-1 - Bridge Chase.mpd:8135 - 8135-1.ldr',
    '8135-1 - Bridge Chase.mpd:8135 - 8135-2.ldr',
    '8135-1 - Bridge Chase.mpd:8135 - 8135-3.ldr',
    '8135-1 - Bridge Chase.mpd:8135 - 8135-4.ldr',
    '8147-1 - Bullet Run.mpd:8147 - 8147-1.ldr',
    '8147-1 - Bullet Run.mpd:8147 - 8147-2.ldr',
    '8147-1 - Bullet Run.mpd:8147 - 8147-5.ldr',
    '8147-1 - Bullet Run.mpd:8147 - 8147-6.ldr',
    '8148-1 - EZ-Roadster - Combination with 8151.mpd',
    '8148-1 - EZ-Roadster.mpd',
    '8149-1 - Midnight Streak - Combination with 8150.mpd',
    '8149-1 - Midnight Streak.mpd',
    '8150-1 - ZX Turbo.mpd',
    '8151-1 - Adrift Sport.mpd',
    '8152-1 - Speed Chasing.mpd:8152 - 8152-1.ldr',
    '8152-1 - Speed Chasing.mpd:8152 - 8152-2.ldr',
    '8152-1 - Speed Chasing.mpd:8152 - 8152-3.ldr',
    '8152-1 - Speed Chasing.mpd:8152 - 8152-4.ldr',
    '8152-1 - Speed Chasing.mpd:8152 - 8152-5.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-1.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-2.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-3.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-4.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-5.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-6.ldr',
    '8154-1 - Brick Street Customs.mpd:8154 - 8154-8.ldr',
    '8182-1 - Monster Crushers.mpd:8182 - 8182-1.ldr',
    '8182-1 - Monster Crushers.mpd:8182 - 8182-2.ldr',
    '8182-1 - Monster Crushers.mpd:8182 - 8182-3.ldr',
    '8182-1 - Monster Crushers.mpd:8182 - 8182-4.ldr',
    '8182-1 - Monster Crushers.mpd:8182 - 8182-5.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-1.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-2.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-3.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-4.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-5.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-6.ldr',
    '8186-1 - Street Extreme.mpd:8186 - 8186-7.ldr',
    '8192-1 - Lime Racer - Combination with 8194.mpd',
    '8192-1 - Lime Racer.mpd',
    '8193-1 - Blue Bullet - Combination with 8195.mpd',
    '8193-1 - Blue Bullet.mpd',
    '8194-1 - Nitro Muscle.mpd',
    '8195-1 - Turbo Tow.mpd',
    '8196-1 - Chopper Jump.mpd:8196 - 8196-1.ldr',
    '8196-1 - Chopper Jump.mpd:8196 - 8196-2.ldr',
    '8197-1 - Highway Chaos.mpd:8197 - 8197-1.ldr',
    '8197-1 - Highway Chaos.mpd:8197 - 8197-2.ldr',
    '8198-1 - Ramp Crash.mpd:8198 - 8198-1.ldr',
    '8198-1 - Ramp Crash.mpd:8198 - 8198-2.ldr',
    '8199-1 - Security Smash.mpd:8199 - 8199-1.ldr',
    '8199-1 - Security Smash.mpd:8199 - 8199-2.ldr',
    '8211-1 - Brick Street Getaway.mpd:8211 - 8211-1.ldr',
    '8211-1 - Brick Street Getaway.mpd:8211 - 8211-2.ldr',
    '8211-1 - Brick Street Getaway.mpd:8211 - 8211-3.ldr',
    '8211-1 - Brick Street Getaway.mpd:8211 - 8211-4.ldr',
    '8211-1 - Brick Street Getaway.mpd:8211 - 8211-5.ldr',
    '8301-1 - Urban Enforcer - Combination with 8304.mpd',
    '8301-1 - Urban Enforcer.mpd',
    '8302-1 - Rod Rider - Combination with 8303.mpd',
    '8302-1 - Rod Rider.mpd',
    '8303-1 - Demon Destroyer.mpd',
    '8304-1 - Smokin\' Slickster.mpd',
    '8641-1 - Flame Glider - Combination with 8642.mpd',
    '8641-1 - Flame Glider.mpd',
    '8642-1 - Monster Crusher.mpd',
    '8643-1 - Power Cruiser - Combination with 8644.ldr',
    '8643-1 - Power Cruiser.mpd',
    '8644-1 - Street Maniac.mpd',
    '8655-1 - RX-Sprinter - Combination with 8658.mpd',
    '8655-1 - RX-Sprinter.mpd',
    '8656-1 - F6 Truck - Combination with 8657.mpd',
    '8656-1 - F6 Truck.mpd',
    '8657-1 - ATR 4.mpd',
    '8658-1 - Big Bling Wheelie.mpd',
    '8661-1 - Carbon Star - Combination with 8666.mpd',
    '8661-1 - Carbon Star.mpd',
    '8662-1 - Blue Renegade - Combination with 8665.mpd',
    '8662-1 - Blue Renegade.mpd',
    '8663-1 - Fat Trax - Combination with 8664.mpd',
    '8663-1 - Fat Trax.mpd',
    '8664-1 - Road Hero.mpd',
    '8665-1 - Highway Enforcer %7BBox%7D.mpd',
    '8666-1 - TunerX.mpd',
    '8681-1 - Tuner Garage.mpd:8681 - 8681-1.ldr',
    '8681-1 - Tuner Garage.mpd:8681 - 8681-2.ldr',
    '8681-1 - Tuner Garage.mpd:8681 - 8681-3.ldr',
    '8681-1 - Tuner Garage.mpd:8681 - 8681-4.ldr',
    '8681-1 - Tuner Garage.mpd:8681 - 8681-5.ldr',
    'Black Racer.mpd',
]

print(len(dataset_paths))

#build_dataset('small_vehicles', settings.paths['omr'], dataset_paths, 24)
