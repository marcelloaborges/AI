using System;

namespace UpperConfidenceBound
{
    class Program
    {
        static void Main(string[] args)
        {
            var m1 = 0.7d;
            var m2 = 0.15d;
            var m3 = 0.3d;

            var bandits = new Bandit[]
            {
                new Bandit(m1),
                new Bandit(m2),
                new Bandit(m3)
            };
            
            var wins = 0;            

            for (int i = 0; i < 1000; i++)
            {
                Bandit upperBandit = null;
                var maxUpperBound = 0d;

                foreach (var bandit in bandits)
                {
                    var upperBound = 0d;

                    if (bandit.Pulls > 0)
                    {
                        var avarageReward = (double)bandit.Wins / bandit.Pulls;
                        var deltaI = Math.Sqrt(1.5d * Math.Log(i + 1) / bandit.Pulls);
                        upperBound = avarageReward + deltaI;
                    }
                    else
                        upperBound = 1e4;

                    if (upperBound > maxUpperBound)
                    {
                        maxUpperBound = upperBound;
                        upperBandit = bandit;
                    }
                }

                Console.WriteLine(upperBandit.RealMean);
                var result = upperBandit.Pull();
                if (result) wins++;
            }

            Console.WriteLine($"B1: Real:{bandits[0].RealMean} - Picks:{bandits[0].Pulls} - Wins:{bandits[0].Wins}");
            Console.WriteLine($"B2: Real:{bandits[1].RealMean} - Picks:{bandits[1].Pulls} - Wins:{bandits[1].Wins}");
            Console.WriteLine($"B3: Real:{bandits[2].RealMean} - Picks:{bandits[2].Pulls} - Wins:{bandits[2].Wins}");
            Console.WriteLine($"Wins: {wins}");
        }
    }

    public class Bandit
    {
        private double _realMean;
        public double RealMean => _realMean;

        private int _pulls;
        public int Pulls => _pulls;

        private int _wins;
        public int Wins => _wins;

        private readonly Random _random;

        public Bandit(double realMean)
        {
            _realMean = realMean;

            _random = new Random();
        }

        public bool Pull()
        {
            var value = _random.NextDouble();

            _pulls++;
            //Console.WriteLine($"{_realMean} pulled");

            if (value < _realMean)
            {
                _wins++;
                return true;
            }

            return false;
        }
    }
}