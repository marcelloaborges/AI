namespace Coach
{
    public class RGBDto
    {
        public int R { get; set; }
        public int G { get; set; }
        public int B { get; set; }
        public Color Color { get; set; }
    }

    public enum Color
    {
        WHITE,
        BLACK,
        RED, 
        GREEN,
        BLUE,
        YELLOW
    }
}