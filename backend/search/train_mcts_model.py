import pandas as pd
import torch

features = torch.tensor(
    pd.read_csv("/home/ubuntu/github/LandsharkGame/mcts/build/features.csv").values
).float()
payoffs = torch.tensor(
    pd.read_csv("/home/ubuntu/github/LandsharkGame/mcts/build/labels.csv").values
).float()
mcts_policy = torch.tensor(
    pd.read_csv("/home/ubuntu/github/LandsharkGame/mcts/build/policy.csv").values
).float()
winners = payoffs.argmax(dim=1, keepdim=True)

dataset = torch.utils.data.TensorDataset(features, winners, mcts_policy)
train_examples = (features.size()[0] * 9) // 10
train_ds, val_ds = torch.utils.data.random_split(
    dataset, (train_examples, features.size()[0] - train_examples)
)
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=128, shuffle=True, pin_memory=True
)
val_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, pin_memory=True)


class Policy(torch.nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, num_players: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.num_players = num_players
        self.shared = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(feature_dim),
                (torch.nn.Linear(feature_dim, 128)),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(128),
                (torch.nn.Linear(128, 64)),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(64),
            ]
        )

        self.critic = torch.nn.ModuleList(
            [torch.nn.Linear(64, num_players), torch.nn.Softmax(dim=1)]
        )

        self.actor = torch.nn.ModuleList(
            [torch.nn.Linear(64, action_dim), torch.nn.Softmax(dim=1)]
        )

        self.critic_criterion = torch.nn.NLLLoss()

        self.optimizer = torch.optim.Adam(
            list(self.shared.parameters())
            + list(self.critic.parameters())
            + list(self.actor.parameters()),
            lr=0.001,
        )

    def forward(self, features: torch.Tensor):
        x = features
        for m in self.shared:
            x = m(x)
        head = x
        for m in self.critic:
            x = m(x)
        critic_output = x
        x = head
        for m in self.actor:
            x = m(x)
        actor_output = x
        return critic_output, actor_output

    @torch.jit.ignore
    def fit(self, train_dl, val_dl):
        previous_loss = None
        stale_count = 0
        train_loss = 0
        train_count = 0
        for x, batch in enumerate(train_dl):
            features, winners, mcts_policy = batch
            features = features.cuda()
            winners = winners.cuda()
            mcts_policy = mcts_policy.cuda()
            self.optimizer.zero_grad()
            total_loss = self.get_loss(features, winners, mcts_policy)
            train_loss += total_loss
            train_count += 1
            total_loss.backward()
            """
            if x > 0 and total_loss + 1e-4 > previous_loss:
                stale_count += 1
                if stale_count >= 3:
                    break
            else:
                stale_count = 0
            previous_loss = float(total_loss)
            """
            self.optimizer.step()
            self.optimizer.zero_grad()
        train_loss /= train_count
        print(f"Train loss: {train_loss}")

        val_loss = 0
        val_count = 0
        for batch in val_dl:
            features, winners, mcts_policy = batch
            features = features.cuda()
            winners = winners.cuda()
            mcts_policy = mcts_policy.cuda()
            total_loss = self.get_loss(features, winners, mcts_policy)
            val_loss += total_loss
            val_count += 1
        val_loss /= val_count
        print("Validation loss:", val_loss)
        return val_loss

    def get_loss(self, features, winners, mcts_policy):
        critic_output, actor_output = self(features.detach())

        critic_loss = self.critic_criterion(
            torch.log(critic_output.clamp(min=1e-3)), winners.flatten()
        )
        actor_loss = (
            -(mcts_policy * torch.log(actor_output.clamp(min=1e-3))).sum(dim=1).mean()
        )
        # print(actor_output)
        # print(mcts_policy)

        total_loss = critic_loss + actor_loss
        assert torch.isnan(actor_loss).sum() == 0
        assert torch.isnan(critic_loss).sum() == 0

        return total_loss


try:
    policy = torch.load("checkpoint_mcts_policy.pt")
except:
    print("Could not load policy, starting over")
    policy = Policy(features.size()[1], mcts_policy.size()[1], 4).cuda()

previous_val_loss = None
fail_count = 0
for x in range(100):
    val_loss = policy.fit(train_dl, val_dl)
    if x > 0:
        if val_loss > (previous_val_loss - 1e-2):
            fail_count += 1
            if fail_count >= 3:
                break
        else:
            fail_count = 0
    previous_val_loss = val_loss

torch.save(policy, "checkpoint_mcts_policy.pt")
script_module = torch.jit.trace(
    policy.cpu().eval(), torch.randn((1, features.size()[1]))
)
script_module.save("trained_mcts_policy.pt")
