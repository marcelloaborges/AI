using System;

namespace Test2
{
    //APPROACH USING MEAN
    class Program
    {
        static void Main(string[] args)
        {
            var m1 = 0.15d;
            var m2 = 0.15d;
            var m3 = 0.30d;
            var em = 0.05d;

            var bandit1 = new Bandit(m1);
            var bandit2 = new Bandit(m2);
            var bandit3 = new Bandit(m3);
            var explore = new Explore(em);

            var random = new Random();
            var wins = 0;
            for (int i = 0; i < 1000; i++)
            {
                var shot = explore.Pull();
                var result = false;

                if (shot)
                {
                    var chosedBandit = random.Next(1, 4);

                    switch (chosedBandit)
                    {
                        case 1:
                            result = bandit1.Pull();
                            bandit1.Update(result);

                            break;
                        case 2:
                            result = bandit2.Pull();
                            bandit2.Update(result);

                            break;
                        case 3:
                            result = bandit3.Pull();
                            bandit3.Update(result);

                            break;
                    }
                }
                else
                {
                    var biggestChance = bandit1.Mean;
                    var chosedBandit = bandit1;

                    if (bandit2.Mean > biggestChance)
                    {
                        chosedBandit = bandit2;
                        biggestChance = bandit2.Mean;
                    }

                    if (bandit3.Mean > biggestChance)
                    {
                        chosedBandit = bandit3;
                        biggestChance = bandit3.Mean;
                    }

                    result = chosedBandit.Pull();
                    chosedBandit.Update(result);
                }

                if (result) wins++;
            }

            Console.WriteLine($"B1: Real:{bandit1.RealMean} - Estimate:{bandit1.Mean}");
            Console.WriteLine($"B2: Real:{bandit2.RealMean} - Estimate:{bandit2.Mean}");
            Console.WriteLine($"B3: Real:{bandit3.RealMean} - Estimate:{bandit3.Mean}");
            Console.WriteLine($"Explore: Estimate:{explore.Mean}");
            Console.WriteLine($"Wins: {wins}");
        }
    }

    public class Explore
    {
        private double _mean;
        public double Mean => _mean;
        private double _n;
        private double _win;
        private Random _random;

        public Explore(double initialMean = 0.5d)
        {
            _mean = initialMean;

            _random = new Random();
        }

        public bool Pull()
        {
            var value = _random.NextDouble();

            if (value < _mean) return true;
            else return false;
        }

        public void Update(bool win)
        {
            _n += 1;
            if (win) _win += 1;

            _mean = _win / _n;
        }
    }

    public class Bandit
    {
        private double _realMean;
        public double RealMean => _realMean;
        private double _mean;
        public double Mean => _mean;
        private double _n;
        private double _win;
        private Random _random;

        public Bandit(double realMean)
        {
            _realMean = realMean;

            _random = new Random();
            _mean = 1;
        }

        public bool Pull()
        {
            var value = _random.NextDouble();

            if (value < _realMean) return true;
            else return false;
        }

        public void Update(bool win)
        {
            _n += 1;
            if (win) _win += 1;

            _mean = _win / _n;
        }
    }
}
